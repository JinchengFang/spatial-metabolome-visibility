#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--ion-set", default="strict", choices=["all", "strict", "protonated"])
ap.add_argument("--n-null", type=int, default=99); ap.add_argument("--block-um", type=float, default=1000.0)
ap.add_argument("--sections-dir", default=None, help="default 02_processed/ravi_maldi_sections"); ap.add_argument("--grid-um", type=float, default=100.0)
ap.add_argument("--include-uncertain", action="store_true"); ap.add_argument("--min-px", type=int, default=3)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step4")
plt.rcParams.update({"font.size": 7}); t0 = time.time(); rng = np.random.default_rng(0); summary = {}
ALPHAS = np.logspace(-2, 4, 13)


def cvpred(Xf, y, folds):
    yh = np.empty(len(y))
    for f in np.unique(folds):
        te = folds == f
        yh[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xf[~te], y[~te]).predict(Xf[te])
    return yh


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def poly2(u, v):
    u = (u - u.mean()) / (u.std() + 1e-9); v = (v - v.mean()) / (v.std() + 1e-9)
    return np.column_stack([np.ones_like(u), u, v, u * u, v * v, u * v])


def detrend(M, T):
    M = np.asarray(M, float); M = M[:, None] if M.ndim == 1 else M
    return M - T @ np.linalg.lstsq(T, M, rcond=None)[0]


def metab_matrix(Xi, var):
    """Ion intensities (TIC-normalized) -> strict ion set -> log1p and z per ion -> mean per metabolite."""
    m = C.ion_mask(var, a.ion_set); names = var.best_name.astype(str).values[m]
    L = np.log1p(np.asarray(Xi)[:, m]); Z = (L - L.mean(0)) / (L.std(0) + 1e-9)
    ML = sorted(set(names))
    return pd.DataFrame({k: Z[:, names == k].mean(1) for k in ML})


# ============================================================== (A) whole-transcriptome test in patients with confirmed identity
IDT, allowed = C.ravi_identity(P, a.include_uncertain)
rows = []
for pid in sorted(allowed):
    f = os.path.join(P["PROC"], "ravi_paired", f"{pid}_visium.h5ad"); fv = os.path.join(P["PROC"], "ravi_paired", f"{pid}_metab_var.tsv")
    if not (os.path.exists(f) and os.path.exists(fv)):
        C.log(f"[A] {pid}: paired files missing, skipped"); continue
    B = ad.read_h5ad(f); var = pd.read_csv(fv, sep="\t", index_col=0)
    ok = B.obs.has_maldi.values.astype(bool) & (B.obs.n_maldi_px.values >= a.min_px)
    B = B[ok].copy()
    Mt = metab_matrix(B.obsm["metab"], var)
    pcols = [c for c in B.obs if c.startswith("prog_")]
    Xc = B.obs[pcols].values
    Xg = B.X.toarray() if sp.issparse(B.X) else np.asarray(B.X)
    hv = np.argsort(-Xg.var(0))[:2000]; Gp = PCA(30, random_state=0).fit_transform(StandardScaler().fit_transform(Xg[:, hv]))
    sxy = B.obsm["spatial_um"]; blk = a.block_um
    bid = (np.floor(sxy[:, 1] / blk) * 1000 + np.floor(sxy[:, 0] / blk)).astype(int); ub = np.unique(bid); bix = np.searchsorted(ub, bid)
    folds = np.random.default_rng(1).permutation(np.arange(len(ub)) % 5)[bix]
    blocks = [np.where(bix == b)[0] for b in range(len(ub))]


    def bperm(Mx):
        order = rng.permutation(len(ub)); out = np.empty_like(Mx)
        for b in range(len(ub)):
            out[blocks[b]] = Mx[np.resize(blocks[order[b]], len(blocks[b]))]
        return out


    C.log(f"[A] {pid}: {B.n_obs} spots (>= {a.min_px} MALDI pixels), {Mt.shape[1]} metabolites, {len(pcols)} programmes + 30 transcriptome PCs; {len(ub)} spatial blocks")
    for level in ("full", "detrended"):
        T = poly2(sxy[:, 0], sxy[:, 1])
        Xc_ = Xc if level == "full" else detrend(Xc, T); Gp_ = Gp if level == "full" else detrend(Gp, T)
        NULLG = [bperm(Gp_) for _ in range(a.n_null)]
        for m in Mt:
            y = Mt[m].values if level == "full" else detrend(Mt[m].values, T)[:, 0]
            rc = r2(y, cvpred(Xc_, y, folds)); rcg = r2(y, cvpred(np.column_stack([Xc_, Gp_]), y, folds)); dr = rcg - rc
            nul = np.array([r2(y, cvpred(np.column_stack([Xc_, Gn]), y, folds)) - rc for Gn in NULLG])
            rows.append(dict(patient=pid, level=level, metabolite=m, n_spots=len(y), R2_composition=rc, R2_comp_genes=rcg, dR2_genes=dr,
                             null_dR2_p95=float(np.percentile(nul, 95)), p_perm=(np.sum(nul >= dr) + 1) / (len(nul) + 1)))
        d_ = pd.DataFrame([r for r in rows if r["patient"] == pid and r["level"] == level])
        C.log(f"    [{level}] median R² composition {d_.R2_composition.median():+.3f}; median delta R²(transcriptome | composition) {d_.dR2_genes.median():+.3f} "
              f"(median null 95th percentile {d_.null_dR2_p95.median():+.3f}); p < 0.05 {int((d_.p_perm < 0.05).sum())}/{len(d_)}")
GA = pd.DataFrame(rows)
if len(GA):
    GA["q"] = GA.groupby(["patient", "level"]).p_perm.transform(lambda v: C.bh(v.values))
    GA.to_csv(os.path.join(TAB, "m6_gbm_beyond_composition.tsv"), sep="\t", index=False)
    summary["A"] = {f"{p}_{lv}": dict(median_R2_comp=float(d.R2_composition.median()), median_dR2=float(d.dR2_genes.median()),
                                       n_sig=int((d.q < 0.05).sum()), n=len(d)) for (p, lv), d in GA.groupby(["patient", "level"])}
else:
    C.log("[A] no GBM patient with high-confidence identity; skipped")

# ============================================================== (B) identity-independent: conservation of metabolite covariation across patients
sdir = a.sections_dir or os.path.join(P["PROC"], "ravi_maldi_sections")
secs = sorted(glob.glob(os.path.join(sdir, "S*.h5ad")))
CM = {}
for f in secs:
    A = ad.read_h5ad(f); X = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
    t = X.sum(1); X = X / np.maximum(t, 1e-12)[:, None] * np.median(t[t > 0])
    xu, yu = A.obs.x_local_um.values, A.obs.y_local_um.values
    gx = np.floor((xu - xu.min()) / a.grid_um).astype(int); gy = np.floor((yu - yu.min()) / a.grid_um).astype(int); nx = gx.max() + 1
    lin_ = gy * nx + gx; ub, inv = np.unique(lin_, return_inverse=True); cnt = np.bincount(inv)
    Bm = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(ub), len(inv)))
    Xg = np.asarray((Bm @ sp.csr_matrix(X)).todense()) / cnt[:, None]
    psz = float(A.uns.get("pixel_size_um", 0) or 0)
    if psz <= 0:   # pixel spacing estimated from coordinates if metadata are missing
        from scipy.spatial import cKDTree
        psz = float(np.median(cKDTree(np.column_stack([xu, yu])).query(np.column_stack([xu, yu])[:5000], k=2)[0][:, 1]))
    keep = cnt >= max(1, (a.grid_um / psz) ** 2 / 2)
    Mt = metab_matrix(Xg[keep], A.var)
    ax_ = np.bincount(inv, weights=A.obs.x_px.values.astype(float)) / cnt; ay_ = np.bincount(inv, weights=A.obs.y_px.values.astype(float)) / cnt
    T = np.column_stack([poly2((ub % nx).astype(float)[keep], (ub // nx).astype(float)[keep]), poly2(ax_[keep], ay_[keep])[:, 1:]])
    CM[os.path.basename(f)[:-5]] = dict(raw=Mt.corr(method="spearman"),
                                        detrended=pd.DataFrame(detrend(Mt.values, T), columns=Mt.columns).corr(method="spearman"), n=int(keep.sum()))
    C.log(f"[B] {os.path.basename(f)[:-5]}: {int(keep.sum())} bins of {a.grid_um:.0f} µm, {Mt.shape[1]} metabolites")
rows, pair_rows = [], []
if len(CM) >= 2:
    common = sorted(set.intersection(*[set(v["raw"].columns) for v in CM.values()]))
    iu = np.triu_indices(len(common), 1)
    for level in ("raw", "detrended"):
        V = {k: v[level].loc[common, common].values[iu] for k, v in CM.items()}
        ks = list(V)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                r0 = spearmanr(V[ks[i]], V[ks[j]])[0]; Mi = CM[ks[i]][level].loc[common, common].values; Mj = CM[ks[j]][level].loc[common, common].values
                nul = []
                for _ in range(499):
                    pm = rng.permutation(len(common)); nul.append(spearmanr(Mi[iu], Mj[np.ix_(pm, pm)][iu])[0])
                rows.append(dict(level=level, sec1=ks[i], sec2=ks[j], n_metab=len(common), corr_of_corr=r0, p_label_perm=(np.sum(np.array(nul) >= r0) + 1) / 500))
        Vm = np.vstack([V[k] for k in ks])
        for t_, (p_, q_) in enumerate(zip(*iu)):
            col = Vm[:, t_]; sg = np.sign(np.median(col))
            pair_rows.append(dict(level=level, m1=common[p_], m2=common[q_], median_r=float(np.median(col)), n_same_sign=int((np.sign(col) == sg).sum()),
                                  n_sections=len(ks), conserved=bool((np.sign(col) == sg).sum() >= len(ks) - 1 and abs(np.median(col)) >= 0.3)))
CC = pd.DataFrame(rows); PRS = pd.DataFrame(pair_rows)
CC.to_csv(os.path.join(TAB, "m6_gbm_covariation_conservation.tsv"), sep="\t", index=False)
if len(PRS):
    # comparison with IDH-mutant (metabolite correlations on the whole-section 200-µm grid; shared metabolites)
    gi = os.path.join(P["PROC"], f"m4_grid_IDHmut_{'' if a.ion_set == 'all' else a.ion_set + '_'}200um.parquet")
    if os.path.exists(gi):
        Gd = pd.read_parquet(gi); Mi = Gd[[c for c in Gd if c.startswith("m:")]]; Mi.columns = [c[2:] for c in Mi.columns]
        Ci = Mi.corr(method="spearman")
        PRS["idhmut_r"] = [Ci.loc[r.m1, r.m2] if r.m1 in Ci and r.m2 in Ci else np.nan for r in PRS.itertuples()]
    PRS = PRS.sort_values(["level", "conserved", "median_r"], ascending=[True, False, False], key=lambda s: s.abs() if s.dtype.kind == "f" else s)
    PRS.to_csv(os.path.join(TAB, "m6_gbm_conserved_pairs.tsv"), sep="\t", index=False)
for lv, d in CC.groupby("level"):
    pr = PRS[PRS.level == lv]
    sh = pr.dropna(subset=["idhmut_r"]) if "idhmut_r" in pr else pd.DataFrame()
    agree = int((np.sign(sh.median_r) == np.sign(sh.idhmut_r)).sum()) if len(sh) else 0
    summary[f"B_{lv}"] = dict(median_corr_of_corr=float(d.corr_of_corr.median()), n_pairs_sig=int((d.p_label_perm < 0.05).sum()), n_section_pairs=len(d),
                              n_conserved_metabolite_pairs=int(pr.conserved.sum()), n_metabolite_pairs=len(pr),
                              idhmut_shared_pairs=len(sh), idhmut_same_sign=agree)
    C.log(f"[B, {lv}] median agreement of correlation matrices between sections rho = {d.corr_of_corr.median():.2f} ({int((d.p_label_perm < 0.05).sum())}/{len(d)} pairs p < 0.05); "
          f"conserved metabolite pairs {int(pr.conserved.sum())}/{len(pr)}; same sign as IDH-mutant for shared pairs {agree}/{len(sh)}")
    for x in pr[pr.conserved].head(8).itertuples():
        C.log(f"     {x.m1} ~ {x.m2}: median r = {x.median_r:+.2f} ({x.n_same_sign}/{x.n_sections} same sign)" + (f"; IDH-mut r = {x.idhmut_r:+.2f}" if "idhmut_r" in pr and np.isfinite(x.idhmut_r) else ""))
if len(CC):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    for axx, lv in zip(axs, ("raw", "detrended")):
        d = CC[CC.level == lv]; ks = sorted(set(d.sec1) | set(d.sec2)); Mx = pd.DataFrame(np.eye(len(ks)), index=ks, columns=ks)
        for r in d.itertuples():
            Mx.loc[r.sec1, r.sec2] = Mx.loc[r.sec2, r.sec1] = r.corr_of_corr
        im = axx.imshow(Mx.values, cmap="viridis", vmin=0, vmax=1); plt.colorbar(im, ax=axx, shrink=.7)
        axx.set_xticks(range(len(ks))); axx.set_xticklabels(ks); axx.set_yticks(range(len(ks))); axx.set_yticklabels(ks)
        axx.set_title(f"GBM: metabolite co-variation similarity ({lv})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m6_gbm_conservation.png"), dpi=160); plt.close(fig)
C.save_json(dict(summary, runtime_s=round(time.time() - t0)), os.path.join(TAB, "m6_04_summary.json"))
C.log(f"done ({time.time() - t0:.0f} s)")
