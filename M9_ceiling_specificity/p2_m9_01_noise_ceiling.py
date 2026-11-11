#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="IDHmut,ccRCC,GBM"); ap.add_argument("--res", default="200,400")
ap.add_argument("--max-feat", type=int, default=60); ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--cluster-r", type=float, default=0.9)
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--ion-set", default="strict"); ap.add_argument("--ccrcc-dir", default=None)
ap.add_argument("--gmt", default=None); ap.add_argument("--include-uncertain", action="store_true"); ap.add_argument("--min-bins", type=int, default=80)
ap.add_argument("--min-px-half", type=int, default=2, help="minimum pixels per bin in each half")
ap.add_argument("--n-pc-other", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-save-bins", action="store_true")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step7")
BINDIR = os.path.join(P["PROC"], "m9_bins"); os.makedirs(BINDIR, exist_ok=True)
RES = [int(r) for r in a.res.split(",")]; DS = [d.strip() for d in a.datasets.split(",") if d.strip()]
rng = np.random.default_rng(a.seed); t0 = time.time(); plt.rcParams.update({"font.size": 7})
SIGDIR = os.path.join(P["PROC"], "signatures")
gmt = a.gmt or next(iter(sorted(glob.glob(os.path.join(P["REFDB"], "h.all.*.symbols.gmt")))), None)
HALL = C.read_gmt(gmt) if gmt and os.path.exists(gmt) else None
STATE_FALLBACK = {k: v for k, v in {**C.PROGRAMS, **C.METAB_PROGRAMS}.items()
                  if k not in ("WM_oligodendrocyte", "Neuron_cortex", "Microglia", "Macrophage_TAM", "Vascular", "T_cell")}
LM = pd.read_csv(os.path.join(P["REFDB"], "_parsed_LIPIDMAPS.tsv.gz"), sep="\t") if os.path.exists(os.path.join(P["REFDB"], "_parsed_LIPIDMAPS.tsv.gz")) else None
C.log(f"state gene sets: {'Hallmark ' + str(len(HALL)) if HALL else 'programmes of p2_common (no Hallmark .gmt)'}")


def sig_of(ds):
    f = os.path.join(SIGDIR, f"{ds}_signature.tsv")
    return pd.read_csv(f, sep="\t", index_col=0) if os.path.exists(f) else None


def tic_norm(X, tic):
    tic = np.asarray(tic, float); f = np.median(tic[tic > 0]) / np.maximum(tic, 1e-12)
    return (sp.diags(f) @ sp.csr_matrix(X)).tocsr() if sp.issparse(X) else X * f[:, None]


# ------------------------------------------------------------------ input: every sample as pixels (µm coordinates, intensities, checkerboard half) + transcriptomic units (spots / cells)
def load_idhmut():
    pix = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"))
    pix = pix[~pix.obs["empty"].values.astype(bool)].copy(); var = pix.var.copy()
    fann = os.path.join(P["TAB"], "sbiad1426_peak_annotation.tsv")
    if os.path.exists(fann):
        ann = pd.read_csv(fann, sep="\t").set_index("peak")
        for c_ in ("tier", "use_downstream", "best_name", "best_adduct", "isotope_of", "ambiguous"):
            if c_ in ann:
                var[c_] = ann.reindex(var.index)[c_].values
    X = pix.X.tocsr() if sp.issparse(pix.X) else sp.csr_matrix(pix.X); X = X.astype(np.float64)
    use = var.use_downstream.fillna(False).astype(bool).values
    tic = np.asarray(X[:, np.where(use)[0]].sum(1)).ravel(); ok = tic > 0
    sel = np.asarray(C.ion_mask(var, a.ion_set), bool)
    Xs = tic_norm(X[ok][:, np.where(sel)[0]], tic[ok])
    names = var.best_name.astype(str).values[sel]
    cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"))
    anno = cel.obs["Anno"].astype(str).values; lin = cel.obs["lineage"].astype(str).values
    vc = pd.Series(anno).value_counts(); ctype = np.where(pd.Series(anno).map(vc).values >= 100, anno, lin)
    G = cel.X.tocsr() if sp.issparse(cel.X) else sp.csr_matrix(cel.X)
    half = ((pix.obs.x_px.values[ok] + pix.obs.y_px.values[ok]) % 2).astype(int)
    return dict(dataset="IDH-mut", sample="SBIAD1426", px_xy=np.column_stack([pix.obs.x_um.values[ok], pix.obs.y_um.values[ok]]), px_X=Xs,
                px_half=half, px_tic=tic[ok], feat_group=names, names=np.array(sorted(set(names))), mz=None, pol="positive", annotated=True,
                kind="xenium", tx_xy=np.asarray(cel.obsm["spatial"], float), counts=G, genes=np.asarray(cel.var_names.astype(str)), ctype=ctype,
                sig=None, programs=None, state_sets=STATE_FALLBACK, state_min=2)


def align_ccrcc(ps, js, JX, SX, s):
    """The SM pixel coordinates of SpatialMETA can be shifted relative to the spot coordinates of the joint object (Y7: about 120 µm).
    With the metabolite values assigned to spots by the data authors as reference, the shift maximizing the median per-feature correlation of spot vs mean of the 4 nearest pixels is searched."""
    sel = np.arange(0, JX.shape[1], max(1, JX.shape[1] // 80))
    def score(off):
        _, i = cKDTree(ps + off).query(js, k=4); agg = SX[i][:, :, sel].mean(1); J_ = JX[:, sel]
        ok = (J_.std(0) > 0) & (agg.std(0) > 0)
        return float(np.nanmedian([np.corrcoef(J_[:, j], agg[:, j])[0, 1] for j in np.where(ok)[0]]))
    best = (score(np.zeros(2)), 0.0, 0.0)
    for step, span, c in ((150, 600, (0.0, 0.0)), (30, 150, None)):
        cx, cy = (best[1], best[2]) if c is None else c
        for dx in np.arange(-span, span + 1, step):
            for dy in np.arange(-span, span + 1, step):
                sc = score(np.array([cx + dx, cy + dy]))
                if sc > best[0]:
                    best = (sc, cx + dx, cy + dy)
    return best


def load_ccrcc():
    d = a.ccrcc_dir or os.path.join(P["PUB"], "SpatialMETA2025_ccRCC_ST_SM"); out = []; arows = []
    for f in sorted(glob.glob(os.path.join(d, "adata_joint_*_raw.h5ad"))):
        s = os.path.basename(f).replace("adata_joint_", "").replace("_raw.h5ad", ""); fsm = os.path.join(d, f"adata_SM_{s}_raw.h5ad")
        if not os.path.exists(fsm):
            C.log(f"[ccRCC {s}] pixel-level file {os.path.basename(fsm)} missing; skipped (the noise ceiling needs pixels)"); continue
        J = ad.read_h5ad(f); typ = pd.Series(J.var["type"].astype(str).values)
        sm, st = typ.str.startswith("SM").values, (typ == "ST").values
        Xj = J.X.tocsc(); JX = Xj[:, np.where(sm)[0]].toarray().astype(float); CT = Xj[:, np.where(st)[0]].tocsr()
        S_ = ad.read_h5ad(fsm); nm = J.var_names[sm].astype(str)
        common = [n for n in nm if n in set(S_.var_names.astype(str))]
        SX = S_[:, common].X; SX = SX.toarray() if sp.issparse(SX) else np.asarray(SX, float)
        JX = JX[:, [list(nm).index(n) for n in common]]
        pol = np.array([{"SM_pos": "positive", "SM_neg": "negative"}.get(t, "unknown") for t in S_[:, common].var["type"].astype(str)], dtype=object)
        ps = np.asarray(S_.obsm["spatial"], float); js = np.asarray(J.obsm["spatial"], float)
        sc, dx, dy = align_ccrcc(ps, js, JX, SX, s); pitch = C.spot_pitch(js); um = 100.0 / pitch
        arows.append(dict(sample=s, fidelity_median_r=sc, offset_x_um=dx * um, offset_y_um=dy * um))
        C.log(f"[ccRCC {s}] pixel-spot alignment: shift ({dx * um:+.0f}, {dy * um:+.0f}) µm, median per-feature correlation {sc:.2f}" + ("  WARNING: < 0.5, alignment doubtful" if sc < 0.5 else ""))
        tic = np.asarray(S_.obs["total_intensity"].values, float) if "total_intensity" in S_.obs else SX.sum(1)
        ok = tic > 0; keep_s = np.asarray(CT.sum(1)).ravel() > 0
        half = ((S_.obs.x_coord_original.values + S_.obs.y_coord_original.values) % 2).astype(int)
        gn = J.var["name"].astype(str).values[st] if "name" in J.var else J.var_names[st].astype(str).values
        out.append(dict(dataset="ccRCC", sample=s, px_xy=(ps[ok] + [dx, dy]) * um, px_X=tic_norm(SX[ok], tic[ok]), px_half=half[ok], px_tic=tic[ok],
                        feat_group=None, names=np.asarray(common), mz=pd.to_numeric(pd.Series(common), errors="coerce").values, pol=pol, annotated=False,
                        kind="visium", tx_xy=js[keep_s] * um, counts=CT[np.where(keep_s)[0]], genes=gn, ctype=None,
                        sig=sig_of("ccRCC"), programs=C.CCRCC_PROGRAMS, state_sets=HALL or STATE_FALLBACK, state_min=10 if HALL else 3))
    if arows:
        pd.DataFrame(arows).to_csv(os.path.join(TAB, "m9_ccrcc_alignment.tsv"), sep="\t", index=False)
    return out


def load_gbm():
    IDT, allowed = C.ravi_identity(P, a.include_uncertain); out = []
    for pid in sorted(allowed):
        fv = os.path.join(P["PROC"], "ravi_paired", f"{pid}_visium.h5ad"); fm = os.path.join(P["PROC"], "ravi_paired", f"{pid}_maldi.h5ad")
        if not (os.path.exists(fv) and os.path.exists(fm)):
            continue
        B = ad.read_h5ad(fv); M = ad.read_h5ad(fm)
        if "tissue" in M.obs:
            M = M[M.obs.tissue.values.astype(bool)].copy()
        X = M.X.toarray() if sp.issparse(M.X) else np.asarray(M.X, float)
        fvt = os.path.join(P["PROC"], "ravi_paired", f"{pid}_metab_var.tsv")
        var = pd.read_csv(fvt, sep="\t", index_col=0) if os.path.exists(fvt) else M.var.copy()
        var.index = var.index.astype(str); var = var.reindex(M.var_names.astype(str))
        tic = X.sum(1); ok = tic > 0; m = np.asarray(C.ion_mask(var, a.ion_set), bool)
        CT = B.layers["counts"] if "counts" in B.layers else sp.csr_matrix(np.expm1(B.X.toarray() if sp.issparse(B.X) else B.X))
        CT = CT.tocsr() if sp.issparse(CT) else sp.csr_matrix(CT)
        half = ((M.obs.x_px.values + M.obs.y_px.values) % 2).astype(int)
        nmz = (var.best_name.astype(str) + " " + var.best_adduct.astype(str)).values[m]
        out.append(dict(dataset="GBM", sample=pid, px_xy=np.column_stack([M.obs.x_vis_um.values, M.obs.y_vis_um.values])[ok],
                        px_X=tic_norm(X[ok][:, m], tic[ok]), px_half=half[ok], px_tic=tic[ok], feat_group=None, names=nmz,
                        mz=(var.mz_cal if "mz_cal" in var else var.mz).values[m], pol=np.array(["negative"] * m.sum()), annotated=True,
                        kind="visium", tx_xy=np.asarray(B.obsm["spatial_um"], float), counts=CT, genes=np.asarray(B.var_names.astype(str)), ctype=None,
                        sig=sig_of("GBM"), programs=C.PROGRAMS, state_sets=HALL or STATE_FALLBACK, state_min=10 if HALL else 3))
    return out


# ------------------------------------------------------------------ binning
def bin_sample(S, R):
    o = np.minimum(S["px_xy"].min(0), S["tx_xy"].min(0))
    kp = np.floor((S["px_xy"] - o) / R).astype(np.int64); kt = np.floor((S["tx_xy"] - o) / R).astype(np.int64)
    keyp = kp[:, 1] * 100000 + kp[:, 0]; keyt = kt[:, 1] * 100000 + kt[:, 0]
    nA = pd.Series(S["px_half"] == 0).groupby(keyp).sum(); nB = pd.Series(S["px_half"] == 1).groupby(keyp).sum()
    nt = pd.Series(1, index=keyt).groupby(level=0).sum()
    mint = 1 if S["kind"] == "visium" else max(1, int(3 * (R / 100.0) ** 2 * 0.5))
    good = sorted(set(nA[(nA >= a.min_px_half) & (nB.reindex(nA.index).fillna(0) >= a.min_px_half)].index) & set(nt[nt >= mint].index))
    gi = {k: i for i, k in enumerate(good)}; nb = len(good)
    if nb < a.min_bins:
        return None
    ip = np.array([gi.get(k, -1) for k in keyp]); it = np.array([gi.get(k, -1) for k in keyt])
    ctr = np.column_stack([(np.array(good) % 100000 + .5) * R, (np.array(good) // 100000 + .5) * R]) + o

    def pix_mean(mask):
        v = mask & (ip >= 0); Bm = sp.csr_matrix((np.ones(v.sum()), (ip[v], np.where(v)[0])), shape=(nb, len(ip)))
        cnt = np.asarray(Bm.sum(1)).ravel(); I = Bm @ S["px_X"]; I = I.toarray() if sp.issparse(I) else np.asarray(I)
        return np.log1p(I / np.maximum(cnt, 1)[:, None])
    allm = np.ones(len(ip), bool)
    Yf, YA, YB = pix_mean(allm), pix_mean(S["px_half"] == 0), pix_mean(S["px_half"] == 1)
    if S["feat_group"] is not None:          # IDH-mut: ions -> metabolites (mean of z over ions of the same metabolite, as in M4-01 / Step 6)
        grp = S["feat_group"]
        Yf, YA, YB = [np.column_stack([C.zcols(Y_)[:, grp == n].mean(1) for n in S["names"]]) for Y_ in (Yf, YA, YB)]
    v = it >= 0; Bt = sp.csr_matrix((np.ones(v.sum()), (it[v], np.where(v)[0])), shape=(nb, len(it)))
    CB = (Bt @ S["counts"]).tocsr(); ncell = np.asarray(Bt.sum(1)).ravel()
    comp_fixed = None
    if S["ctype"] is not None:
        ct = S["ctype"][v]; types = sorted(set(ct))
        comp_fixed = np.column_stack([np.bincount(it[v][ct == t], minlength=nb) for t in types]) / np.maximum(ncell, 1)[:, None]
    return dict(xy=ctr, Yf=Yf, YA=YA, YB=YB, CB=CB, ncell=ncell, comp_fixed=comp_fixed,
                npx=np.bincount(ip[ip >= 0], minlength=nb), px_tic_bin=np.bincount(ip[ip >= 0], weights=S["px_tic"][ip >= 0], minlength=nb))


def blocks_for(S, CB, ncell, comp_fixed):
    extra = np.log1p(ncell) if S["kind"] == "xenium" else None
    B, LG, src = C.tx_blocks_from_bins(CB, S["genes"], sig=S["sig"], programs_fallback=S["programs"], state_sets=S["state_sets"],
                                       state_min_genes=S["state_min"], comp_fixed=comp_fixed, tech_extra=extra)
    return B, LG, src


def r2(Y, X, folds):
    return C.r2_columns(Y, C.cv_ridge_gcv(X, Y, folds))


def other_pcs(Ysrc, j, npc):
    oth = np.delete(Ysrc, j, axis=1)
    return PCA(min(npc, oth.shape[1]), random_state=0).fit_transform(oth) if oth.shape[1] > npc else oth


# ------------------------------------------------------------------ main loop
samples = []
if "IDHmut" in DS:
    try:
        samples.append(load_idhmut())
    except FileNotFoundError as e:
        C.log(f"[IDH-mut] skipped: {e}")
if "ccRCC" in DS:
    samples += load_ccrcc()
if "GBM" in DS:
    samples += load_gbm()
rows = []
for S in samples:
    for R in RES:
        Bn = bin_sample(S, R)
        if Bn is None:
            C.log(f"  {S['dataset']} {S['sample']} @ {R} µm: too few bins; skipped"); continue
        Yf0 = Bn["Yf"]
        det = (np.expm1(Yf0) > 0).mean(0) if S["feat_group"] is None else np.ones(Yf0.shape[1])
        okf = np.where(det >= a.min_det)[0]
        Lf = C.zcols(Yf0[:, okf])
        reps, I, q, members = C.select_structured_features(Lf, Bn["xy"], a.max_feat, a.cluster_r, seed=R)
        if len(reps) < 3:
            C.log(f"  {S['dataset']} {S['sample']} @ {R} µm: fewer than 3 structured features; skipped"); continue
        names = np.asarray(S["names"])[okf]
        mzs = np.asarray(S["mz"], float)[okf] if S["mz"] is not None else np.full(len(okf), np.nan)
        pols = (np.asarray(S["pol"], dtype=object)[okf] if not isinstance(S["pol"], str) else np.array([S["pol"]] * len(okf), dtype=object))
        ann = names.astype(str) if S["annotated"] else np.array([C.annotate_mz(m_, p_, LM) for m_, p_ in zip(mzs, pols)], dtype=object)
        # transcriptome: full depth + two binomially thinned halves
        Bf, LG, src = blocks_for(S, Bn["CB"], Bn["ncell"], Bn["comp_fixed"])
        CA, CBh = C.thin_counts(Bn["CB"], rng)
        BA, _, _ = blocks_for(S, CA, Bn["ncell"], Bn["comp_fixed"]); BB, _, _ = blocks_for(S, CBh, Bn["ncell"], Bn["comp_fixed"])
        folds = C.spatial_folds(Bn["xy"], a.block_um, seed=R)
        C.log(f"  {S['dataset']} {S['sample']} @ {R} µm: {len(Bn['xy'])} bins (median {np.median(Bn['npx']):.0f} pixels per bin); {len(reps)} features after merging; composition = {src}")
        if not a.no_save_bins:
            ge = (LG > 0).mean(0) >= 0.05
            np.savez_compressed(os.path.join(BINDIR, f"{S['dataset']}_{S['sample']}_{R}um.npz"), xy=Bn["xy"], Yf=Yf0[:, okf].astype(np.float32),
                                YA=Bn["YA"][:, okf].astype(np.float32), YB=Bn["YB"][:, okf].astype(np.float32), names=names.astype(str), mz=mzs, pol=pols.astype(str),
                                annotation=np.asarray(ann).astype(str), moran_I=I, moran_q=q, reps=reps, LG=LG[:, ge].astype(np.float32),
                                genes=np.asarray(S["genes"])[ge].astype(str), comp=np.asarray(Bf["comp"], np.float32), tech=np.asarray(Bf["tech"], np.float32),
                                comp_src=src, dataset=S["dataset"], sample=S["sample"], res=R)
        for level in ("full", "detrended"):
            T = C.poly2(Bn["xy"])
            prep = (lambda M_: C.zcols(M_)) if level == "full" else (lambda M_: C.zcols(C.detrend(M_, T)))
            Yf = prep(Yf0[:, okf][:, reps]); YA = prep(Bn["YA"][:, okf][:, reps]); YB = prep(Bn["YB"][:, okf][:, reps])
            def X_of(Bk):
                M_ = np.column_stack([np.asarray(v_, float) for v_ in Bk.values()])
                M_ = M_ if level == "full" else C.detrend(M_, T)
                return M_[:, M_.std(0) > 1e-9]
            Xf, XA, XB = X_of(Bf), X_of(BA), X_of(BB)
            rAB, SB = C.split_half_reliability(YA, YB)
            R_tx_full = r2(Yf, Xf, folds)
            R_tx_half = (r2(YA, Xf, folds) + r2(YB, Xf, folds)) / 2
            R_tx_thin = (r2(Yf, XA, folds) + r2(Yf, XB, folds)) / 2
            R_m_full = np.empty(len(reps)); R_m_same = np.empty(len(reps)); R_m_cross = np.empty(len(reps))
            for j in range(len(reps)):
                OA, OB, Of = other_pcs(YA, j, a.n_pc_other), other_pcs(YB, j, a.n_pc_other), other_pcs(Yf, j, a.n_pc_other)
                R_m_full[j] = r2(Yf[:, [j]], Of, folds)[0]
                R_m_same[j] = (r2(YA[:, [j]], OA, folds)[0] + r2(YB[:, [j]], OB, folds)[0]) / 2
                R_m_cross[j] = (r2(YB[:, [j]], OA, folds)[0] + r2(YA[:, [j]], OB, folds)[0]) / 2
            # depth extrapolation: R²(d) = R²inf d/(d+k); k (0 <= k <= 3) from the half- and full-depth points
            ratio = np.where((R_tx_thin > 0.01) & (R_tx_full > 0.01), R_tx_full / np.maximum(R_tx_thin, 1e-9), np.nan)
            k = np.clip((ratio - 1) / np.maximum(2 - ratio, 1e-3), 0, 3); k = np.where(ratio >= 2, 3, k)
            R_tx_inf = np.where(np.isfinite(k), R_tx_full * (1 + k), R_tx_full)
            R_tx_inf = np.minimum(R_tx_inf, np.maximum(np.nan_to_num(SB, nan=1.0), R_tx_full))     # extrapolation capped at the noise ceiling
            ceil_h = np.where(rAB > 0.05, rAB, np.nan); ceil_f = np.where(SB > 0.1, SB, np.nan)
            for jj, col in enumerate(reps):
                rows.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=R, level=level, feature=names[col], mz=mzs[col], polarity=pols[col],
                                 annotation=ann[col], moran_I=float(I[col]), n_bins=len(Yf), composition_source=src,
                                 r_AB=rAB[jj], reliability_SB=SB[jj], R2_tx_full=R_tx_full[jj], R2_tx_halfY=R_tx_half[jj], R2_tx_thin=R_tx_thin[jj],
                                 R2_tx_inf=R_tx_inf[jj], R2_maldi_full=R_m_full[jj], R2_maldi_same_half=R_m_same[jj], R2_maldi_cross_half=R_m_cross[jj],
                                 tx_full_norm=R_tx_full[jj] / ceil_f[jj], tx_inf_norm=R_tx_inf[jj] / ceil_f[jj], tx_half_norm=R_tx_half[jj] / ceil_h[jj],
                                 maldi_cross_norm=R_m_cross[jj] / ceil_h[jj], maldi_same_norm=R_m_same[jj] / ceil_h[jj]))
            d_ = pd.DataFrame([r for r in rows if r["dataset"] == S["dataset"] and r["sample"] == S["sample"] and r["res_um"] == R and r["level"] == level])
            C.log(f"     [{level}] median reliability SB {d_.reliability_SB.median():.2f} (r_AB {d_.r_AB.median():.2f}); R²_tx full {d_.R2_tx_full.median():+.3f} "
                  f"(half depth {d_.R2_tx_thin.median():+.3f} -> extrapolated {d_.R2_tx_inf.median():+.3f}); R²_tx / ceiling {d_.tx_full_norm.median():.2f}; "
                  f"MALDI same half {d_.R2_maldi_same_half.median():+.3f} / cross half {d_.R2_maldi_cross_half.median():+.3f} (/ceiling {d_.maldi_cross_norm.median():.2f})")
if not rows:
    raise SystemExit("no results (check inputs)")
D = pd.DataFrame(rows)
D["gap_norm"] = D.maldi_cross_norm - D.tx_half_norm
D["reliable"] = D.reliability_SB >= 0.7
D.to_csv(os.path.join(TAB, "m9_ceiling.tsv"), sep="\t", index=False)


def summ(g):
    r = g[g.reliable]
    return pd.Series(dict(n=len(g), n_reliable=len(r), SB=g.reliability_SB.median(), r_AB=g.r_AB.median(), frac_reliable=g.reliable.mean(),
                          R2_tx_full=g.R2_tx_full.median(), R2_tx_thin=g.R2_tx_thin.median(), R2_tx_inf=g.R2_tx_inf.median(),
                          depth_gain=(g.R2_tx_inf - g.R2_tx_full).median(), R2_maldi_full=g.R2_maldi_full.median(),
                          R2_maldi_same=g.R2_maldi_same_half.median(), R2_maldi_cross=g.R2_maldi_cross_half.median(),
                          shared_pixel_noise=(g.R2_maldi_same_half - g.R2_maldi_cross_half).median(),
                          tx_norm_reliable=r.tx_full_norm.median() if len(r) else np.nan, tx_inf_norm_reliable=r.tx_inf_norm.median() if len(r) else np.nan,
                          maldi_cross_norm=g.maldi_cross_norm.median(), gap_norm=g.gap_norm.median(),
                          n_tx_norm_gt_0_5=int((r.tx_full_norm > 0.5).sum()) if len(r) else 0))


SM = D.groupby(["dataset", "sample", "res_um", "level"]).apply(summ).reset_index()
SM["D30_gap_not_noise"] = SM.tx_norm_reliable < 0.5
SM["D31_depth_not_limiting"] = SM.depth_gain < 0.05
SM["D32_internal_exceeds_tx"] = SM.gap_norm >= 0.2
SM.to_csv(os.path.join(TAB, "m9_ceiling_summary.tsv"), sep="\t", index=False)
C.log("summary (median):\n" + SM.round(3).to_string(index=False))

# figure: per sample, ceiling / transcriptome (full depth, extrapolated) / MALDI cross-half; right: per-feature R²_tx vs reliability
R0 = 200 if 200 in RES else RES[0]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1.4, 1]))
for axx, lv in zip(axs[:2], ("full", "detrended")):
    e = SM[(SM.res_um == R0) & (SM.level == lv)].reset_index(drop=True); x = np.arange(len(e))
    axx.bar(x - .3, e.SB, .2, color="#d9d9d9", label="noise ceiling (split-half SB)")
    axx.bar(x - .1, e.R2_tx_full, .2, color="#de2d26", label="transcriptome (full depth)")
    axx.scatter(x - .1, e.R2_tx_inf, marker="_", s=70, c="k", zorder=3, label="transcriptome (∞ depth)")
    axx.bar(x + .1, e.R2_maldi_cross, .2, color="#636363", label="other metabolites (cross-half)")
    axx.bar(x + .3, e.R2_maldi_same, .2, color="#bdbdbd", hatch="//", label="other metabolites (same pixels)")
    axx.set_xticks(x); axx.set_xticklabels([f"{r.dataset}\n{r['sample']}" for _, r in e.iterrows()], fontsize=5.5)
    axx.axhline(0, c="k", lw=.4); axx.set_ylabel("median cross-validated R²"); axx.set_title(f"{R0} µm · {lv}"); axx.legend(fontsize=5, frameon=False)
d = D[(D.res_um == R0) & (D.level == "full")]
for ds, c in zip(sorted(d.dataset.unique()), ("#3182bd", "#e6550d", "#31a354", "#756bb1")):
    e = d[d.dataset == ds]; axs[2].scatter(e.reliability_SB, e.R2_tx_full, s=6, alpha=.6, c=c, label=ds)
axs[2].plot([0, 1], [0, 1], "k--", lw=.6); axs[2].plot([0, 1], [0, .5], ":", c="grey", lw=.6)
axs[2].set_xlabel("metabolite reliability (noise ceiling)"); axs[2].set_ylabel("transcriptome → metabolite R²"); axs[2].set_xlim(0, 1); axs[2].set_ylim(-.1, 1)
axs[2].legend(fontsize=6, frameon=False); axs[2].set_title("each metabolite cluster (full)")
fig.suptitle("Is the transcriptome–metabolome gap a measurement artefact?", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(os.path.join(FIG, "m9_ceiling.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
