#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="ccRCC,GBM"); ap.add_argument("--res", default="100,200,400")
ap.add_argument("--n-null", type=int, default=49); ap.add_argument("--max-feat", type=int, default=100); ap.add_argument("--min-det", type=float, default=0.10)
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--ccrcc-dir", default=None); ap.add_argument("--ion-set", default="strict")
ap.add_argument("--include-uncertain", action="store_true"); ap.add_argument("--min-bins", type=int, default=80)
ap.add_argument("--n-power", type=int, default=5, help="repetitions of the power simulation (signal added to all features at once)"); ap.add_argument("--effects", default="0.02,0.05")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step5")
plt.rcParams.update({"font.size": 7}); t0 = time.time(); rng = np.random.default_rng(0)
RES = [int(r) for r in a.res.split(",")]; DS = [d.strip() for d in a.datasets.split(",") if d.strip()]
ALPHAS = np.logspace(-2, 4, 13)


# ------------------------------------------------------------------ loaders: dict(sample, xy_um, SM (spot x feature, TIC-normalized), sm_names, mz, pol, counts (spot x gene), genes, programs)
def load_ccrcc():
    d = a.ccrcc_dir or os.path.join(P["PUB"], "SpatialMETA2025_ccRCC_ST_SM")
    fs = sorted(glob.glob(os.path.join(d, "adata_joint_*_raw.h5ad")))
    if not fs:
        C.log(f"[ccRCC] {d}/adata_joint_*_raw.h5ad not found (run p2_step5_downloads.sh first); skipped"); return []
    out = []
    for f in fs:
        s = os.path.basename(f).replace("adata_joint_", "").replace("_raw.h5ad", "")
        J = ad.read_h5ad(f); typ = pd.Series(J.var["type"].astype(str).values)
        sm, st = typ.str.startswith("SM").values, (typ == "ST").values
        X = J.X.tocsc() if sp.issparse(J.X) else sp.csc_matrix(J.X)
        SM = X[:, np.where(sm)[0]].toarray().astype(float); CT = X[:, np.where(st)[0]].tocsr()
        names = J.var_names[sm].astype(str).values
        pol = np.array(["unknown"] * len(names), dtype=object)
        fsm = os.path.join(d, f"adata_SM_{s}_raw.h5ad")
        if os.path.exists(fsm):
            S_ = ad.read_h5ad(fsm, backed="r"); mp = dict(zip(S_.var_names.astype(str), S_.var["type"].astype(str))); S_.file.close()
            pol = np.array([{"SM_pos": "positive", "SM_neg": "negative"}.get(mp.get(n, ""), "unknown") for n in names], dtype=object)
        mz = pd.to_numeric(pd.Series(names).str.extract(r"([0-9]+\.?[0-9]*)")[0], errors="coerce").values
        tic = SM.sum(1); keep = (tic > 0) & (np.asarray(CT.sum(1)).ravel() > 0)
        SM = SM[keep] / tic[keep, None] * np.median(tic[keep])
        xy = np.asarray(J.obsm["spatial"], float)[keep]; xy = xy * 100.0 / C.spot_pitch(xy)
        gn = J.var["name"].astype(str).values[st] if "name" in J.var else J.var_names[st].astype(str).values
        out.append(dict(dataset="ccRCC", sample=s, xy_um=xy, SM=SM, sm_names=names, mz=mz, pol=pol, counts=CT[keep], genes=pd.Index(gn), programs=C.CCRCC_PROGRAMS))
        C.log(f"[ccRCC] {s}: {keep.sum()} spots, {len(names)} metabolite features (positive {int((pol == 'positive').sum())} / negative {int((pol == 'negative').sum())}), {len(gn)} genes")
    return out


def load_gbm():
    IDT, allowed = C.ravi_identity(P, a.include_uncertain); out = []
    for pid in sorted(allowed):
        f = os.path.join(P["PROC"], "ravi_paired", f"{pid}_visium.h5ad"); fv = os.path.join(P["PROC"], "ravi_paired", f"{pid}_metab_var.tsv")
        if not (os.path.exists(f) and os.path.exists(fv)):
            continue
        B = ad.read_h5ad(f); var = pd.read_csv(fv, sep="\t", index_col=0)
        ok = B.obs.has_maldi.values.astype(bool) & (B.obs.n_maldi_px.values >= 1)
        m = C.ion_mask(var, a.ion_set)
        SM = np.asarray(B.obsm["metab"])[ok][:, m].astype(float)
        names = (var.best_name.astype(str) + " " + var.best_adduct.astype(str)).values[m]
        CT = B.layers["counts"] if "counts" in B.layers else sp.csr_matrix(np.expm1(B.X.toarray() if sp.issparse(B.X) else B.X))
        CT = (CT.tocsr() if sp.issparse(CT) else sp.csr_matrix(CT))[np.where(ok)[0]]
        out.append(dict(dataset="GBM", sample=pid, xy_um=np.asarray(B.obsm["spatial_um"], float)[ok], SM=SM, sm_names=names, mz=var.mz_cal.values[m] if "mz_cal" in var else var.mz.values[m],
                        pol=np.array(["negative"] * m.sum(), dtype=object), counts=CT, genes=pd.Index(B.var_names.astype(str)), programs=C.PROGRAMS, annotated=True))
        C.log(f"[GBM] {pid}: {ok.sum()} spots with MALDI, {m.sum()} strict ions, {B.n_vars} genes")
    if not out:
        C.log("[GBM] no patient with high-confidence identity; skipped")
    return out


def salt_mask(mz, pol):
    bad = np.zeros(len(mz), bool)
    for p_ in ("positive", "negative"):
        sel = (pol == p_) & np.isfinite(mz)
        if sel.any():
            ions = C.inorganic_cluster_ions(p_, 1200).mz.values
            d = np.abs(mz[sel][:, None] - ions[None, :]) / mz[sel][:, None] * 1e6
            bad[np.where(sel)[0]] = (d < 5).any(1)
    return bad


def cv_multi(X, Y, folds):
    Yh = np.empty_like(Y)
    for f in np.unique(folds):
        te = folds == f; sc = StandardScaler().fit(X[~te])
        Yh[te] = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(X[~te]), Y[~te]).predict(sc.transform(X[te])).reshape(-1, Y.shape[1])
    return Yh


def r2_cols(Y, Yh):
    return 1 - ((Y - Yh) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def poly2(xy):
    u = (xy[:, 0] - xy[:, 0].mean()) / xy[:, 0].std(); v = (xy[:, 1] - xy[:, 1].mean()) / xy[:, 1].std()
    return np.column_stack([np.ones_like(u), u, v, u * u, v * v, u * v])


def detrend(M, T):
    return M - T @ np.linalg.lstsq(T, M, rcond=None)[0]


samples = (load_ccrcc() if "ccRCC" in DS else []) + (load_gbm() if "GBM" in DS else [])
if not samples:
    raise SystemExit("no samples to analyse")
rows, qrows = [], []
for S in samples:
    for R in RES:
        if R <= 100:
            inv = np.arange(len(S["xy_um"])); ctr = S["xy_um"]
        else:
            inv, ctr = C.bin_spots(S["xy_um"], R)
        nb = inv.max() + 1
        if nb < a.min_bins:
            C.log(f"  {S['sample']} @ {R} µm: only {nb} bins (< {a.min_bins}); skipped"); continue
        Bm = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(nb, len(inv))); cnt = np.asarray(Bm.sum(1)).ravel()
        SMb = (Bm @ S["SM"]) / cnt[:, None]
        GB = (Bm @ S["counts"]).toarray() if sp.issparse(S["counts"]) else Bm @ S["counts"]
        LG = np.log1p(GB / np.maximum(GB.sum(1, keepdims=True), 1) * 1e4)
        det = (SMb > 0).mean(0); okf = det >= a.min_det
        okf &= ~salt_mask(np.asarray(S["mz"], float), S["pol"])
        L = np.log1p(SMb[:, okf]); L = (L - L.mean(0)) / (L.std(0) + 1e-9)
        names = np.asarray(S["sm_names"])[okf]; mzs = np.asarray(S["mz"])[okf]; pols = np.asarray(S["pol"])[okf]
        I, pI = C.moran_knn(L, ctr, k=6, n_perm=99, seed=R)
        qI = C.bh(pI); struct = qI < 0.05
        order = np.argsort(-np.where(struct, I, -np.inf))[: min(a.max_feat, int(struct.sum()))]
        for j in range(len(names)):
            qrows.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=R, feature=names[j], mz=mzs[j], polarity=pols[j], det_frac=float(det[okf][j]),
                              moran_I=float(I[j]), q_moran=float(qI[j]), analysed=bool(j in set(order))))
        if len(order) < 3:
            C.log(f"  {S['sample']} @ {R} µm: fewer than 3 structured features; skipped"); continue
        Yall = L[:, order]
        Comp, used = C.score_programs(LG, S["genes"], S["programs"], min_genes=3)
        Met, _ = C.score_programs(LG, S["genes"], C.METAB_PROGRAMS, min_genes=3)
        hv = np.argsort(-LG.var(0))[:2000]; Gp = PCA(min(30, len(hv) - 1), random_state=0).fit_transform(StandardScaler().fit_transform(LG[:, hv]))
        blk = a.block_um
        bid = (np.floor(ctr[:, 1] / blk) * 1000 + np.floor(ctr[:, 0] / blk)).astype(int); ub, bix = np.unique(bid, return_inverse=True)
        folds = np.random.default_rng(R).permutation(np.arange(len(ub)) % 5)[bix]
        blocks = [np.where(bix == b)[0] for b in range(len(ub))]


        def bperm(M_):
            o = rng.permutation(len(ub)); out = np.empty_like(M_)
            for b in range(len(ub)):
                out[blocks[b]] = M_[np.resize(blocks[o[b]], len(blocks[b]))]
            return out


        C.log(f"  {S['dataset']} {S['sample']} @ {R} µm: {nb} bins, {int(okf.sum())} features passing QC, {int(struct.sum())} structured -> {len(order)} analysed; "
              f"{Comp.shape[1]} composition programmes, {Met.shape[1]} metabolic programmes, {len(ub)} spatial blocks")
        for level in ("full", "detrended"):
            T = poly2(ctr)
            Y = Yall if level == "full" else detrend(Yall, T)
            Xc = Comp.values if level == "full" else detrend(Comp.values, T)
            Xm = Met.values if level == "full" else detrend(Met.values, T)
            Xg = Gp if level == "full" else detrend(Gp, T)
            r_c = r2_cols(Y, cv_multi(Xc, Y, folds))
            r_cm = r2_cols(Y, cv_multi(np.column_stack([Xc, Xm]), Y, folds))
            r_cg = r2_cols(Y, cv_multi(np.column_stack([Xc, Xg]), Y, folds))
            nm_, ng_ = [], []
            for _ in range(a.n_null):
                nm_.append(r2_cols(Y, cv_multi(np.column_stack([Xc, bperm(Xm)]), Y, folds)) - r_c)
                ng_.append(r2_cols(Y, cv_multi(np.column_stack([Xc, bperm(Xg)]), Y, folds)) - r_c)
            nm_, ng_ = np.array(nm_), np.array(ng_)
            dm, dg = r_cm - r_c, r_cg - r_c
            # internal MALDI reference: prediction of the feature from the other features (first 20 PCs, feature itself excluded)
            r_o = np.empty(Y.shape[1])
            for j in range(Y.shape[1]):
                oth = np.delete(Y, j, axis=1); k_ = min(20, oth.shape[1])
                Op = PCA(k_, random_state=0).fit_transform(oth) if oth.shape[1] > k_ else oth
                r_o[j] = r2_cols(Y[:, [j]], cv_multi(Op, Y[:, [j]], folds))[0]
            # power: add a metabolic-programme signal beyond composition (programme residualized on composition) explaining a fraction e of the composition-residual variance
            Mres = Xm - np.column_stack([np.ones(len(Xm)), Xc]) @ np.linalg.lstsq(np.column_stack([np.ones(len(Xm)), Xc]), Xm, rcond=None)[0]
            Mres = (Mres - Mres.mean(0)) / (Mres.std(0) + 1e-9)
            vres = ((Y - cv_multi(Xc, Y, folds)) ** 2).mean(0)
            thr = np.percentile(nm_, 95, 0); pw = {}
            for e in [float(x) for x in a.effects.split(",")]:
                hits = np.zeros(Y.shape[1])
                for _ in range(a.n_power):
                    Sg = Mres[:, rng.integers(Mres.shape[1], size=Y.shape[1])]
                    Ys = Y + Sg * np.sqrt(e * vres / (1 - e))
                    rcs = r2_cols(Ys, cv_multi(Xc, Ys, folds)); dms = r2_cols(Ys, cv_multi(np.column_stack([Xc, Xm]), Ys, folds)) - rcs
                    hits += dms > thr
                pw[e] = hits / a.n_power
            for j in range(Y.shape[1]):
                rows.append(dict(**{f"power_metabprog_{e:g}": float(v[j]) for e, v in pw.items()}, dataset=S["dataset"], sample=S["sample"], res_um=R, level=level, feature=names[order[j]], mz=mzs[order[j]], polarity=pols[order[j]],
                                 moran_I=float(I[order[j]]), n_bins=nb, R2_composition=r_c[j], R2_comp_metabprog=r_cm[j], R2_comp_genes=r_cg[j],
                                 dR2_metabprog=dm[j], null95_metabprog=float(np.percentile(nm_[:, j], 95)), p_metabprog=(np.sum(nm_[:, j] >= dm[j]) + 1) / (a.n_null + 1),
                                 dR2_genes=dg[j], null95_genes=float(np.percentile(ng_[:, j], 95)), p_genes=(np.sum(ng_[:, j] >= dg[j]) + 1) / (a.n_null + 1),
                                 null_med_metabprog=float(np.median(nm_[:, j])), null_med_genes=float(np.median(ng_[:, j])),
                                 R2_other_metabolites=r_o[j]))
            C.log(f"     [{level}] median R² composition {np.median(r_c):+.3f}; delta R² metabolic programmes {np.median(dm):+.3f} (null 95% {np.median(np.percentile(nm_, 95, 0)):+.3f}); "
                  f"delta R² whole transcriptome {np.median(dg):+.3f} (null 95% {np.median(np.percentile(ng_, 95, 0)):+.3f}); internal MALDI {np.median(r_o):+.3f}")
D = pd.DataFrame(rows); Q = pd.DataFrame(qrows)
Q.to_csv(os.path.join(TAB, "m7_feature_qc.tsv"), sep="\t", index=False)
if not len(D):
    raise SystemExit("no results")
for k in ("metabprog", "genes"):
    D[f"q_{k}"] = D.groupby(["dataset", "sample", "res_um", "level"])[f"p_{k}"].transform(lambda v: C.bh(v.values))
    D[f"meaningful_{k}"] = (D[f"q_{k}"] < 0.05) & (D[f"dR2_{k}"] >= 0.02)
# null-corrected increments (removing the overfitting cost of 30 extra predictors); the composition share uses corrected increments
D["dR2_genes_adj"] = D.dR2_genes - D.null_med_genes; D["dR2_metabprog_adj"] = D.dR2_metabprog - D.null_med_metabprog
D["share_composition"] = D.R2_composition.clip(lower=0) / (D.R2_composition.clip(lower=0) + D.dR2_genes_adj.clip(lower=0)).clip(lower=1e-6)
D.to_csv(os.path.join(TAB, "m7_spot_decomposition.tsv"), sep="\t", index=False)
SUM = D.groupby(["dataset", "sample", "res_um", "level"]).agg(
    n_features=("feature", "size"), R2_composition=("R2_composition", "median"), dR2_metabprog=("dR2_metabprog", "median"),
    dR2_genes=("dR2_genes", "median"), dR2_genes_adj=("dR2_genes_adj", "median"), dR2_metabprog_adj=("dR2_metabprog_adj", "median"), null95_genes=("null95_genes", "median"), R2_other=("R2_other_metabolites", "median"),
    frac_meaningful_genes=("meaningful_genes", "mean"), frac_meaningful_metabprog=("meaningful_metabprog", "mean"),
    share_composition=("share_composition", "median"), **{c: (c, "median") for c in D.columns if c.startswith("power_")}).reset_index()
SUM.to_csv(os.path.join(TAB, "m7_spot_summary.tsv"), sep="\t", index=False)
C.log("summary (median across features):\n" + SUM.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for axx, lv in zip(axs, ("full", "detrended")):
    s_ = SUM[SUM.level == lv].copy(); s_["lab"] = s_["dataset"] + " " + s_["sample"] + " · " + s_.res_um.astype(str)
    yy = np.arange(len(s_))
    axx.barh(yy - .2, s_.R2_composition, .2, label="R² composition", color="#3182bd")
    axx.barh(yy, s_.dR2_genes, .2, label="ΔR² whole transcriptome | comp.", color="#de2d26")
    axx.barh(yy + .2, s_.dR2_metabprog, .2, label="ΔR² metabolic programs | comp.", color="#fd8d3c")
    axx.scatter(s_.R2_other, yy, marker="|", c="k", s=60, label="MALDI internal reference", zorder=3)
    axx.set_yticks(yy); axx.set_yticklabels(s_.lab, fontsize=5.5); axx.axvline(0, c="k", lw=.4); axx.set_xlabel("median across structured features")
    axx.set_title(f"Spot-based datasets ({lv})"); axx.legend(fontsize=5.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m7_spot_decomposition.png"), dpi=160); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
