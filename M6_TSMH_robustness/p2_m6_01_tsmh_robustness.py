#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore", category=RuntimeWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--ion-set", default="strict", choices=["all", "strict", "protonated"])
ap.add_argument("--res", type=int, default=200); ap.add_argument("--fine", type=int, default=20, help="fine grid (µm) for the deformation simulation")
ap.add_argument("--n-rep", type=int, default=20); ap.add_argument("--n-perm", type=int, default=199)
ap.add_argument("--tumor-frac", type=float, default=0.5); ap.add_argument("--block-um", type=float, default=1000.0)
ap.add_argument("--disp", default="10,30,50,100", help="RMS displacement of the random deformations (µm)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step4")
plt.rcParams.update({"font.size": 7}); t0 = time.time(); rng = np.random.default_rng(0)
R = a.res; F_ = a.fine; assert R % F_ == 0, "--res must be a multiple of --fine"; K = R // F_
DISP = [float(d) for d in a.disp.split(",")]

# ------------------------------------------------------------------ input (same preprocessing and grid origin as M4-01)
pix = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"))
pix = pix[~pix.obs["empty"].values.astype(bool)].copy()
var = pix.var.copy()
ann = pd.read_csv(os.path.join(P["TAB"], "sbiad1426_peak_annotation.tsv"), sep="\t").set_index("peak")
for c_ in ("tier", "use_downstream", "best_name", "best_adduct", "isotope_of", "ambiguous"):
    if c_ in ann:
        var[c_] = ann.reindex(var.index)[c_].values
X = (pix.X.tocsr() if sp.issparse(pix.X) else sp.csr_matrix(pix.X)).astype(np.float64)
use = var.use_downstream.fillna(False).astype(bool).values
tic = np.asarray(X[:, np.where(use)[0]].sum(1)).ravel(); ok = tic > 0
X = X[ok]; tic = tic[ok]; px = pix.obs.x_um.values[ok]; py = pix.obs.y_um.values[ok]
acq_x = pix.obs.x_px.values[ok].astype(float); acq_y = pix.obs.y_px.values[ok].astype(float)     # acquisition raster (direction of instrument drift)
sel = C.ion_mask(var, a.ion_set); ions = var.index[sel]; metab = var.loc[ions, "best_name"].astype(str).values; ML = sorted(set(metab))
tier = var.tier.astype(str).values
mat_idx = np.where(tier == "matrix/background")[0]; salt_idx = np.where(tier == "inorganic/salt")[0]
cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"))
cx, cy = cel.obsm["spatial"][:, 0], cel.obsm["spatial"][:, 1]
x0, y0 = min(px.min(), cx.min()), min(py.min(), cy.min())
lin = cel.obs["lineage"].astype(str).values
C.log(f"IDH-mut: {len(px)} pixels; ion set '{a.ion_set}': {len(ions)} ions / {len(ML)} metabolites; {len(mat_idx)} matrix ions, {len(salt_idx)} salt-cluster ions")

# ------------------------------------------------------------------ ion maps on the fine grid (F_ µm), one per independent pixel half (checkerboard)
fx = np.floor((px - x0) / F_).astype(int); fy = np.floor((py - y0) / F_).astype(int)
NX = (fx.max() // K + 1) * K; NY = (fy.max() // K + 1) * K
norm = np.median(tic) / tic
half = ((np.floor(px / 10).astype(int) + np.floor(py / 10).astype(int)) % 2).astype(bool)   # checkerboard of 10-µm pixels


def fine_maps(cols, mask):
    """TIC-normalized mean maps (NY, NX, len(cols)) (NaN without pixels) and counts."""
    lin_ = fy[mask] * NX + fx[mask]
    cnt = np.bincount(lin_, minlength=NX * NY).astype(float)
    Xs = sp.diags(norm[mask]) @ X[mask][:, cols]
    B = sp.csr_matrix((np.ones(mask.sum()), (lin_, np.arange(mask.sum()))), shape=(NX * NY, mask.sum()))
    S = np.asarray((B @ Xs).todense()) / np.maximum(cnt, 1)[:, None]
    S[cnt == 0] = np.nan
    return S.reshape(NY, NX, -1), cnt.reshape(NY, NX)


def coarse(img):
    """Fine grid -> R µm (nan-mean over K x K blocks)."""
    sh = img.shape
    v = img.reshape(sh[0] // K, K, sh[1] // K, K, *sh[2:])
    return np.nanmean(v, axis=(1, 3))


ion_cols = np.where(sel)[0]
Fall, Call = fine_maps(ion_cols, np.ones(len(px), bool))
FA, _ = fine_maps(ion_cols, half); FB, _ = fine_maps(ion_cols, ~half)
CNT_R = Call.reshape(NY // K, K, NX // K, K).sum((1, 3))

# ------------------------------------------------------------------ R-µm grid: cell composition, tumour bins, technical covariates
gx_c = np.floor((cx - x0) / R).astype(int); gy_c = np.floor((cy - y0) / R).astype(int); nxr, nyr = NX // K, NY // K
kc = (gx_c < nxr) & (gy_c < nyr)
ncell = np.zeros((nyr, nxr)); np.add.at(ncell, (gy_c[kc], gx_c[kc]), 1)
nmal = np.zeros((nyr, nxr)); np.add.at(nmal, (gy_c[kc & (lin == "Malignant")], gx_c[kc & (lin == "Malignant")]), 1)
fmal = nmal / np.maximum(ncell, 1)
good = (CNT_R >= (R / 10.0) ** 2 / 3) & (ncell >= 3 * (R / 100.0) ** 2 * 0.5)
tum = good & (fmal >= a.tumor_frac)
gy, gx = np.where(tum)
C.log(f"{R} µm: {int(good.sum())} valid bins, {len(gx)} tumour bins")
if len(gx) < 60:
    raise SystemExit("fewer than 60 tumour bins; cannot be analysed")


def metab_z(Fimg):
    """Fine-grid ion maps -> R µm -> log1p -> z of each ion within tumour bins -> mean over ions of the same metabolite. Returns (n_tumor_bins x n_metab) and per-ion matrices."""
    Cimg = np.log1p(coarse(Fimg))[gy, gx]
    Z = (Cimg - np.nanmean(Cimg, 0)) / (np.nanstd(Cimg, 0) + 1e-9)
    Z = np.nan_to_num(Z)
    return np.column_stack([Z[:, metab == m].mean(1) for m in ML]), Z


Mall, Zion = metab_z(Fall)
cov0 = np.column_stack([fmal[gy, gx], np.log1p(ncell[gy, gx])])
Mres = C.residualize_ranks(Mall, cov0)

# technical covariates
tic_img = np.zeros(NY * NX); np.add.at(tic_img, fy * NX + fx, np.log(tic)); tic_img = tic_img.reshape(NY, NX) / np.maximum(Call, 1); tic_img[Call == 0] = np.nan
tech = {"log_TIC": coarse(tic_img)[gy, gx], "pixel_coverage": CNT_R[gy, gx]}
for nm_, idx in (("matrix_ions", mat_idx), ("salt_clusters", salt_idx)):
    if len(idx):
        Fm, _ = fine_maps(idx, np.ones(len(px), bool)); tech[nm_] = np.log1p(coarse(Fm).sum(2))[gy, gx]
# K/Na adduct index: mean of log(K) - log(Na) over metabolites with both [M+K]+ and [M+Na]+
allm = C.ion_mask(var, "all"); add_ = var.best_adduct.fillna("").astype(str).values; nm_all = var.best_name.fillna("").astype(str).values
pairs = [(np.where(allm & (nm_all == m) & (add_ == "[M+K]+"))[0], np.where(allm & (nm_all == m) & (add_ == "[M+Na]+"))[0]) for m in set(nm_all[allm])]
pairs = [(k_[0], n_[0]) for k_, n_ in pairs if len(k_) and len(n_)]
if pairs:
    Fk, _ = fine_maps(np.array([p[0] for p in pairs]), np.ones(len(px), bool)); Fn, _ = fine_maps(np.array([p[1] for p in pairs]), np.ones(len(px), bool))
    tech["K_Na_index"] = np.nanmean(np.log1p(coarse(Fk)) - np.log1p(coarse(Fn)), 2)[gy, gx]
from scipy.ndimage import distance_transform_edt
tech["dist_tissue_edge_um"] = distance_transform_edt(good)[gy, gx] * R
T_ = pd.DataFrame(tech).fillna(0)
# large-scale trends / acquisition drift: quadratic polynomial in bin and acquisition-raster coordinates (biological gradients and instrument drift cannot be separated in one section, hence a separate "local" level)
lin_r = (np.floor((py - y0) / R).astype(int)) * (NX // K) + np.floor((px - x0) / R).astype(int)
cnt_r = np.bincount(lin_r, minlength=(NX // K) * (NY // K))
ax_ = (np.bincount(lin_r, weights=acq_x, minlength=len(cnt_r)) / np.maximum(cnt_r, 1)).reshape(NY // K, NX // K)[gy, gx]
ay_ = (np.bincount(lin_r, weights=acq_y, minlength=len(cnt_r)) / np.maximum(cnt_r, 1)).reshape(NY // K, NX // K)[gy, gx]


def poly2(u, v):
    u = (u - u.mean()) / (u.std() + 1e-9); v = (v - v.mean()) / (v.std() + 1e-9)
    return np.column_stack([u, v, u * u, v * v, u * v])


TREND = np.column_stack([poly2(gx.astype(float), gy.astype(float)), poly2(ax_, ay_)])
COV = {"res": cov0, "local": np.column_stack([cov0, T_.values, TREND])}
C.log(f"technical covariates: {list(T_.columns)} (K/Na index from {len(pairs)} metabolites)")


def moran_perm(vals, n=a.n_perm, X_=None, Y_=None):
    X_ = gx if X_ is None else X_; Y_ = gy if Y_ is None else Y_
    im = np.full((nyr, nxr), np.nan); im[Y_, X_] = vals; I0 = C.morans_i_grid(im); nul = []
    for _ in range(n):
        im[Y_, X_] = rng.permutation(vals); nul.append(C.morans_i_grid(im))
    return I0, (np.sum(np.array(nul) >= I0) + 1) / (n + 1)


# ------------------------------------------------------------------ Q1 technical covariates
rows = []
Mtech = C.residualize_ranks(Mall, np.column_stack([cov0, T_.values]))
Mloc = C.residualize_ranks(Mall, COV["local"])
trend_r2 = 1 - Mloc.var(0) / np.maximum(Mtech.var(0), 1e-12)          # fraction of variance absorbed by the quadratic trend after technical adjustment
for i, m in enumerate(ML):
    I1, p1 = moran_perm(Mres[:, i]); I2, p2 = moran_perm(Mtech[:, i]); I3, p3 = moran_perm(Mloc[:, i])
    r_t = {c_: float(np.corrcoef(Mres[:, i], C._rank_cols(T_[c_].values))[0, 1]) for c_ in T_}
    rows.append(dict(metabolite=m, moran_I=I1, p=p1, moran_I_tech_adj=I2, p_tech_adj=p2, frac_I_retained=I2 / I1 if I1 > 0 else np.nan,
                     moran_I_local=I3, p_local=p3, var_frac_trend=float(trend_r2[i]),
                     r_acq_x=float(np.corrcoef(Mres[:, i], ax_)[0, 1]), r_acq_y=float(np.corrcoef(Mres[:, i], ay_)[0, 1]),
                     max_abs_r_tech=max(abs(v) for v in r_t.values()), worst_tech=max(r_t, key=lambda k: abs(r_t[k]))))
Q1 = pd.DataFrame(rows); Q1["q"] = C.bh(Q1.p.values); Q1["q_tech_adj"] = C.bh(Q1.p_tech_adj.values); Q1["q_local"] = C.bh(Q1.p_local.values)
Q1["structured"] = Q1.q < 0.05; Q1["structured_after_tech"] = Q1.q_tech_adj < 0.05; Q1["structured_local"] = Q1.q_local < 0.05
Q1.to_csv(os.path.join(TAB, "m6_tech_confounders.tsv"), sep="\t", index=False)
# tumour-bin matrices for M6-02: coordinates, residuals on [malignant fraction, density], residuals additionally on technical covariates, technical covariates
pd.concat([pd.DataFrame(dict(gx=gx, gy=gy, fmal=fmal[gy, gx], ncell=ncell[gy, gx])), pd.DataFrame(Mres, columns=["res:" + m for m in ML]),
           pd.DataFrame(Mtech, columns=["tech:" + m for m in ML]), pd.DataFrame(Mloc, columns=["local:" + m for m in ML]),
           T_.add_prefix("cov:").reset_index(drop=True), pd.DataFrame(TREND, columns=[f"trend:{i}" for i in range(TREND.shape[1])])], axis=1) \
    .to_parquet(os.path.join(P["PROC"], f"m6_tumor_bins_{a.ion_set}_{R}um.parquet"))
C.log(f"Q1 structured within tumour: {int(Q1.structured.sum())}/{len(Q1)}; still structured after technical adjustment: {int(Q1.structured_after_tech.sum())} (median retained Moran's I {Q1.frac_I_retained.median():.2f}); "
      f"local structure after removing quadratic trend / drift: {int(Q1.structured_local.sum())}")

# ------------------------------------------------------------------ Q2 adduct coherence
rows = []
Zres = C.residualize_ranks(Zion, cov0)
kna = C._rank_cols(T_["K_Na_index"].values) if "K_Na_index" in T_ else None
for m in ML:
    idx = np.where(metab == m)[0]
    if len(idx) < 2:
        continue
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            r = float(np.corrcoef(Zres[:, idx[i]], Zres[:, idx[j]])[0, 1])
            rk = [abs(float(np.corrcoef(Zres[:, k_], kna)[0, 1])) for k_ in (idx[i], idx[j])] if kna is not None else [np.nan]
            rows.append(dict(metabolite=m, ion1=ions[idx[i]], adduct1=add_[sel][idx[i]], ion2=ions[idx[j]], adduct2=add_[sel][idx[j]],
                             r_within_tumor=r, max_abs_r_with_KNa=max(rk)))
Q2 = pd.DataFrame(rows)
if len(Q2):
    Q2["coherent"] = (Q2.r_within_tumor > 0.3) & (Q2.r_within_tumor > Q2.max_abs_r_with_KNa)
Q2.to_csv(os.path.join(TAB, "m6_adduct_coherence.tsv"), sep="\t", index=False)
if len(Q2):
    C.log(f"Q2 {len(Q2)} adduct pairs ({Q2.metabolite.nunique()} metabolites): median within-tumour residual correlation {Q2.r_within_tumor.median():.2f}; coherent (r > 0.3 and > |r(K/Na)|) {int(Q2.coherent.sum())}")

# ------------------------------------------------------------------ Q4 transcriptomic side: within-tumour predictability of programmes + split-half reliability
GT = f"{'' if a.ion_set == 'all' else a.ion_set + '_'}"
gfile = os.path.join(P["PROC"], f"m4_grid_IDHmut_{GT}{R}um.parquet")
if not os.path.exists(gfile):
    raise SystemExit(f"{gfile} missing; run M4-01 --ion-set {a.ion_set} --step Step4 first")
Gd = pd.read_parquet(gfile)
key = pd.Series(np.arange(len(gx)), index=pd.MultiIndex.from_arrays([gx, gy]))
Gd = Gd[pd.MultiIndex.from_arrays([Gd.gx, Gd.gy]).isin(key.index)].copy()
Gd["_i"] = key.reindex(pd.MultiIndex.from_arrays([Gd.gx, Gd.gy])).values
Gd = Gd.sort_values("_i"); idx_g = Gd._i.values.astype(int)
gcols = [c for c in Gd if c.startswith("g:") and Gd[c].std() > 0]; ccols = [c for c in Gd if c.startswith("ct:")]
pcols = [c for c in Gd if c.startswith("prog:")]
blk = max(3, int(round(a.block_um / R)))
bid = (gy[idx_g] // blk) * 10000 + (gx[idx_g] // blk); ub = np.unique(bid)
folds = np.array([dict(zip(ub, np.random.default_rng(1).permutation(np.arange(len(ub)) % 5)))[b] for b in bid])
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cvr2(Xf, y):
    yh = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        yh[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(Xf[~te], y[~te]).predict(Xf[te])
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


rows = []
genes_used = {p_: [g for g in (C.PROGRAMS.get(p_[5:], "") + " " + C.METAB_PROGRAMS.get(p_[5:], "")).split()] for p_ in pcols}
for p_ in pcols:
    y = C.residualize_ranks(Gd[p_].values, cov0[idx_g])[:, 0]
    other = [c for c in gcols if c[2:] not in genes_used[p_]]
    I_, pI = moran_perm(y, n=99, X_=gx[idx_g], Y_=gy[idx_g])
    rows.append(dict(program=p_[5:], R2_from_other_genes=cvr2(Gd[other].values, y), moran_I=I_, p_moran=pI))
Q4 = pd.DataFrame(rows)
# Xenium split-half reliability (cells of each tumour bin randomly split in two; correlation of programme scores between halves, Spearman-Brown corrected)
G = cel.X.tocsr() if sp.issparse(cel.X) else sp.csr_matrix(cel.X); genes = pd.Index(cel.var_names)
cell_bin = np.where(kc, gy_c * nxr + gx_c, -1); tbin = gy * nxr + gx; pos = pd.Series(np.arange(len(tbin)), index=tbin)
inT = np.isin(cell_bin, tbin); hb = rng.random(len(cx)) < 0.5
SH = []
for h in (True, False):
    m_ = inT & (hb == h); r_ = pos.reindex(cell_bin[m_]).values.astype(int)
    B = sp.csr_matrix((np.ones(m_.sum()), (r_, np.where(m_)[0])), shape=(len(tbin), len(cx)))
    GB = np.asarray((B @ G).todense()); LG = np.log1p(GB / np.maximum(GB.sum(1, keepdims=True), 1) * 1e4)
    S_, _ = C.score_programs(LG, genes, {**C.PROGRAMS, **C.METAB_PROGRAMS}, min_genes=2); SH.append(S_)
rel = {}
for p_ in SH[0].columns:
    r = np.corrcoef(C.residualize_ranks(SH[0][p_].values, cov0)[:, 0], C.residualize_ranks(SH[1][p_].values, cov0)[:, 0])[0, 1]
    rel[p_] = 2 * r / (1 + r) if r > -1 else np.nan
Q4["split_half_reliability"] = Q4.program.map(rel)
Q4.to_csv(os.path.join(TAB, "m6_transcript_internal.tsv"), sep="\t", index=False)
relT = float(np.nanmedian(list(rel.values())))
C.log(f"Q4 transcriptomic side (within tumour): median R² of programmes predicted from the other genes {Q4.R2_from_other_genes.median():.3f}; median split-half reliability {relT:.2f}")

# ------------------------------------------------------------------ Q3 detectability ceiling (random smooth deformations)
def warp_field(rms_um, corr_um=500.0):
    s = corr_um / F_
    u = gaussian_filter(rng.standard_normal((NY, NX)), s); v = gaussian_filter(rng.standard_normal((NY, NX)), s)
    sc = rms_um / F_ / (np.sqrt(np.mean(u ** 2 + v ** 2)) + 1e-12)
    return u * sc, v * sc


def warp(img, u, v):
    yy, xx = np.mgrid[0:NY, 0:NX].astype(float)
    out = np.empty_like(img)
    for j in range(img.shape[2]):
        a_ = img[:, :, j]; msk = np.isfinite(a_)
        w = map_coordinates(np.where(msk, a_, 0), [yy + v, xx + u], order=1, mode="constant", cval=0)
        wm = map_coordinates(msk.astype(float), [yy + v, xx + u], order=1, mode="constant", cval=0)
        out[:, :, j] = np.where(wm > 0.5, w / np.maximum(wm, 1e-9), np.nan)
    return out


ref_d = 50 if 50 in DISP else DISP[len(DISP) // 2]
MA, _ = metab_z(FA); MB0, _ = metab_z(FB)
WARPED = {0.0: [MB0]}
for d in DISP:
    WARPED[d] = []
    for rep in range(a.n_rep):
        u, v = warp_field(d); WARPED[d].append(metab_z(warp(FB, u, v))[0])
Q3s, ceil_by = [], {}
for level in ("res", "local"):
    cv_ = COV[level]; MAr = C.residualize_ranks(MA, cv_)
    ceil = {d: [[float(np.corrcoef(MAr[:, i], C.residualize_ranks(Mw, cv_)[:, i])[0, 1]) for i in range(len(ML))] for Mw in WARPED[d]] for d in WARPED}
    ceil_by[level] = ceil
    C.log(f"  [{level}] median cross-half correlation (deformation RMS 0/" + "/".join(f"{d:.0f}" for d in DISP) + " µm): " + ", ".join(f"{np.median(np.mean(ceil[d], 0)):.2f}" for d in [0.0] + DISP))
    # observed: block cross-validated R² of the transcriptome (composition + all panel genes), best single programme |rho|, R² from other metabolites
    Ml = C.residualize_ranks(Mall, cv_)[idx_g]; cvg = cv_[idx_g]
    rows = []
    for i, m in enumerate(ML):
        oth = [j for j, mm in enumerate(ML) if mm != m]
        row = dict(level=level, metabolite=m, moran_I=Q1.moran_I[i] if level == "res" else Q1.moran_I_local[i],
                   structured=bool(Q1.structured[i] if level == "res" else Q1.structured_local[i]),
                   split_half_r_metab=float(np.mean(ceil[0.0], 0)[i]), R2_obs_transcriptome=cvr2(Gd[ccols + gcols].values, Ml[:, i]),
                   best_single_program_rho=float(np.max(np.abs([np.corrcoef(Ml[:, i], C.residualize_ranks(Gd[p_].values, cvg)[:, 0])[0, 1] for p_ in pcols]))),
                   R2_other_metabolites=cvr2(Ml[:, oth], Ml[:, i]))
        for d in DISP:
            cr = float(np.mean(np.array(ceil[d])[:, i])); row[f"ceiling_R2_disp{int(d)}"] = max(cr, 0) ** 2 * relT
        rows.append(row)
    Qx = pd.DataFrame(rows)
    Qx["detectable"] = Qx[f"ceiling_R2_disp{int(ref_d)}"] >= 0.1
    Qx["silent_despite_detectable"] = Qx.structured & Qx.detectable & (Qx.R2_obs_transcriptome < 0.05)
    st_ = Qx[Qx.structured]
    C.log(f"Q3 [{level}] {len(st_)} structured metabolites: median detectability ceiling R² (deformation {ref_d:.0f} µm x transcriptomic reliability {relT:.2f}) "
          f"{st_[f'ceiling_R2_disp{int(ref_d)}'].median() if len(st_) else float('nan'):.3f}; median observed transcriptomic R² {st_.R2_obs_transcriptome.median() if len(st_) else float('nan'):.3f}; "
          f"detectable but silent: {', '.join(Qx[Qx.silent_despite_detectable].metabolite) or 'none'}")
    Q3s.append(Qx)
Q3 = pd.concat(Q3s, ignore_index=True); Q3.to_csv(os.path.join(TAB, "m6_detectability.tsv"), sep="\t", index=False)
ceil = ceil_by["res"]
Q3r = Q3[Q3.level == "res"].reset_index(drop=True); Q3l = Q3[Q3.level == "local"].reset_index(drop=True)
st = Q3r[Q3r.structured]
core = sorted(set(Q3r[Q3r.silent_despite_detectable].metabolite) & set(Q1[Q1.structured_after_tech].metabolite))
C.log(f"core TSMH metabolites (structured, structured after technical adjustment, detectable but silent): {', '.join(core) or 'none'}")

# ------------------------------------------------------------------ figures
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
s_ = Q3r.sort_values("moran_I", ascending=False)
xx = np.arange(len(s_)); w_ = 0.27
axs[0].bar(xx - w_, s_[f"ceiling_R2_disp{int(ref_d)}"], w_, label=f"detectability ceiling ({ref_d:.0f} µm deformation × RNA reliability)", color="#9ecae1")
axs[0].bar(xx, s_.R2_obs_transcriptome.clip(lower=-0.05), w_, label="observed: transcriptome (comp + panel genes)", color="#de2d26")
axs[0].bar(xx + w_, s_.R2_other_metabolites.clip(lower=-0.05), w_, label="MALDI internal: other metabolites", color="#636363")
axs[0].set_xticks(xx); axs[0].set_xticklabels(s_.metabolite, rotation=90); axs[0].axhline(0, c="k", lw=.4); axs[0].set_ylabel("R² (within tumor)")
axs[0].legend(fontsize=5.5, frameon=False); axs[0].set_title(f"IDH-mut, within tumor ({R} µm): could we have seen it?")
for d in [0.0] + DISP:
    axs[1].plot([d] * len(ML), np.mean(ceil[d], 0), "o", ms=2, alpha=.4, color="grey")
axs[1].plot([0.0] + DISP, [np.median(np.mean(ceil[d], 0)) for d in [0.0] + DISP], "o-", c="#3182bd", lw=2, label="median")
axs[1].axvline(31, ls="--", c="r", lw=.6); axs[1].text(33, .05, "learned misregistration\n(median 31 µm, Step 2)", fontsize=5.5, color="r")
axs[1].set_xlabel("RMS deformation (µm)"); axs[1].set_ylabel("cross-half correlation of metabolite map"); axs[1].set_title("Attenuation by adjacent-section deformation")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m6_detectability.png"), dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.scatter(Q1.moran_I, Q1.moran_I_tech_adj, s=14)
for r_ in Q1.itertuples():
    ax.annotate(r_.metabolite, (r_.moran_I, r_.moran_I_tech_adj), fontsize=5)
lim = [min(0, Q1.moran_I.min()), Q1.moran_I.max() * 1.05]; ax.plot(lim, lim, "k--", lw=.5)
ax.set_xlabel("within-tumor Moran's I"); ax.set_ylabel("after technical covariates"); ax.set_title("Q1: structure is not ionisation / matrix / salt / edge")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m6_tech_confounders.png"), dpi=160); plt.close(fig)
C.save_json(dict(ion_set=a.ion_set, res_um=R, n_tumor_bins=int(len(gx)), n_metabolites=len(ML),
                 Q1_structured=int(Q1.structured.sum()), Q1_structured_after_tech=int(Q1.structured_after_tech.sum()), Q1_median_I_retained=float(Q1.frac_I_retained.median()),
                 Q2_pairs=int(len(Q2)), Q2_coherent=int(Q2.coherent.sum()) if len(Q2) else 0, Q2_median_r=float(Q2.r_within_tumor.median()) if len(Q2) else None,
                 Q3_ref_disp_um=ref_d, rna_reliability=relT, Q3_median_ceiling=float(st[f"ceiling_R2_disp{int(ref_d)}"].median()) if len(st) else None,
                 Q3_median_observed=float(st.R2_obs_transcriptome.median()) if len(st) else None,
                 Q3_median_other_metabolites=float(st.R2_other_metabolites.median()) if len(st) else None,
                 Q3_silent_despite_detectable=sorted(Q3r[Q3r.silent_despite_detectable].metabolite),
                 Q3_local_structured=int(Q3l.structured.sum()), Q3_local_silent_despite_detectable=sorted(Q3l[Q3l.silent_despite_detectable].metabolite),
                 Q1_structured_local=int(Q1.structured_local.sum()), Q1_median_var_frac_trend=float(Q1.var_frac_trend.median()), tsmh_core=core,
                 Q4_median_R2_programs=float(Q4.R2_from_other_genes.median()), runtime_s=round(time.time() - t0)), os.path.join(TAB, "m6_01_summary.json"))
C.log(f"done ({time.time() - t0:.0f} s)")
