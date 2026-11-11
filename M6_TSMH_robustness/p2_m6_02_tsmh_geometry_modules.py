#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore", category=RuntimeWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--ion-set", default="strict", choices=["all", "strict", "protonated"])
ap.add_argument("--res", type=int, default=200); ap.add_argument("--use", default="res", choices=["res", "tech", "local"],
                help="res = residualized on [malignant fraction, density] (primary); tech = also on technical covariates; local = also without quadratic trend / drift (most conservative)")
ap.add_argument("--module-r", type=float, default=0.4); ap.add_argument("--n-perm", type=int, default=1999)
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--radius-um", type=float, default=300.0)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step4")
plt.rcParams.update({"font.size": 7}); t0 = time.time(); R = a.res; rng = np.random.default_rng(0)
f6 = os.path.join(P["PROC"], f"m6_tumor_bins_{a.ion_set}_{R}um.parquet")
if not os.path.exists(f6):
    raise SystemExit(f"{f6} missing; run M6-01 first")
TB = pd.read_parquet(f6); gx, gy = TB.gx.values, TB.gy.values
pre = a.use + ":"
ML = [c[len(pre):] for c in TB if c.startswith(pre)]
M = TB[[pre + m for m in ML]].values
Q1 = pd.read_csv(os.path.join(TAB, "m6_tech_confounders.tsv"), sep="\t").set_index("metabolite")
scol = {"res": "structured", "tech": "structured_after_tech", "local": "structured_local"}[a.use]
struct = [m for m in ML if bool(Q1.loc[m, scol])] if scol in Q1 else ML
C.log(f"{len(gx)} tumour bins ({R} µm); {len(ML)} metabolites, {len(struct)} structured within tumour; residual level '{a.use}'")
if len(struct) < 2:
    C.save_json(dict(ion_set=a.ion_set, res_um=R, use=a.use, n_tumor_bins=int(len(gx)), structured=struct, note="fewer than 2 structured metabolites; module / geometry analysis not applicable"),
                os.path.join(TAB, f"m6_02_summary_{a.use}.json"))
    C.log("fewer than 2 structured metabolites at this residual level; stopping"); sys.exit(0)

# ------------------------------------------------------------------ geometric covariates (same grid origin as M4 / M6-01)
pix = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"), backed="r")
tis = ~pix.obs["empty"].values.astype(bool); px, py = pix.obs.x_um.values[tis], pix.obs.y_um.values[tis]; pix.file.close()
cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"), backed="r")
cxy = cel.obsm["spatial"][:].astype(float); lin = cel.obs["lineage"].astype(str).values; cel.file.close()
x0, y0 = min(px.min(), cxy[:, 0].min()), min(py.min(), cxy[:, 1].min())
ctr = np.column_stack([x0 + (gx + .5) * R, y0 + (gy + .5) * R])
nxr, nyr = int(max(gx.max(), np.floor((cxy[:, 0] - x0) / R).max()) + 2), int(max(gy.max(), np.floor((cxy[:, 1] - y0) / R).max()) + 2)
ix_c = np.floor((cxy[:, 0] - x0) / R).astype(int); iy_c = np.floor((cxy[:, 1] - y0) / R).astype(int)
ncell = np.zeros((nyr, nxr)); np.add.at(ncell, (iy_c, ix_c), 1)


def frac(l_):
    g = np.zeros((nyr, nxr)); s = lin == l_; np.add.at(g, (iy_c[s], ix_c[s]), 1); return g / np.maximum(ncell, 1)


occupied = ncell >= 3
fm, fo, fn = frac("Malignant"), frac("Oligodendrocyte"), frac("Neuron")
geo = {}
vas = cxy[lin == "Vascular"]; mye = cxy[lin == "Myeloid"]
if len(vas):
    tv = cKDTree(vas); geo["log_dist_vessel_um"] = np.log1p(tv.query(ctr)[0])
    geo["vessel_density_r"] = np.array([len(x) for x in tv.query_ball_point(ctr, a.radius_um)])
if len(mye):
    geo["myeloid_density_r"] = np.array([len(x) for x in cKDTree(mye).query_ball_point(ctr, a.radius_um)])
for nm, msk in (("dist_tumor_border_um", occupied & (fm < 0.5)), ("dist_white_matter_um", occupied & (fo >= 0.5)),
                ("dist_neuron_rich_um", occupied & (fn >= 0.3)), ("dist_tissue_edge_um", ~occupied)):
    if msk.any():
        geo[nm] = distance_transform_edt(~msk)[gy, gx] * R
GEO = pd.DataFrame(geo)
C.log(f"geometric covariates: {list(GEO.columns)}")
cov0 = np.column_stack([TB.fmal.values, np.log1p(TB.ncell.values)])
Gres = C.residualize_ranks(GEO.values, cov0)
blk = max(3, int(round(a.block_um / R)))

# ------------------------------------------------------------------ 1. modules
S = [ML.index(m) for m in struct]
Cm = np.corrcoef(M[:, S].T) if len(S) > 1 else np.ones((1, 1))
if len(S) >= 3:
    Z_ = linkage(squareform(1 - Cm, checks=False), "average")
    lab = fcluster(Z_, t=1 - a.module_r, criterion="distance")
else:
    lab = np.arange(1, len(S) + 1)
mods = {}
for k in np.unique(lab):
    mem = [struct[i] for i in np.where(lab == k)[0]]
    mods[f"MOD{len(mods) + 1}" if len(mem) > 1 else f"single:{mem[0]}"] = mem
MODT = pd.DataFrame([dict(module=k, n=len(v), members=", ".join(v),
                          mean_within_r=float(np.mean([Cm[struct.index(x), struct.index(y)] for i, x in enumerate(v) for y in v[i + 1:]])) if len(v) > 1 else np.nan)
                     for k, v in mods.items()])
MS = pd.DataFrame({k: M[:, [ML.index(m) for m in v]].mean(1) for k, v in mods.items()})
MODT.to_csv(os.path.join(TAB, f"m6_modules_{a.use}.tsv"), sep="\t", index=False)
C.log("metabolite modules: " + "; ".join(f"{r.module} [{r.members}] (within-module r {r.mean_within_r:.2f})" for r in MODT[MODT.n > 1].itertuples()))
# reproducibility: two spatial halves
rows = []
for split, coord in (("left_vs_right", gx), ("top_vs_bottom", gy)):
    h = coord <= np.median(coord)
    if min(h.sum(), (~h).sum()) < 30 or len(S) < 3:
        continue
    A1 = np.corrcoef(M[h][:, S].T); A2 = np.corrcoef(M[~h][:, S].T); iu = np.triu_indices(len(S), 1)
    r0 = spearmanr(A1[iu], A2[iu])[0]; nul = []
    for _ in range(999):
        pmt = rng.permutation(len(S)); nul.append(spearmanr(A1[iu], A2[np.ix_(pmt, pmt)][iu])[0])
    # fraction of within-module pairs with the same sign
    same = [np.sign(A1[struct.index(x), struct.index(y)]) == np.sign(A2[struct.index(x), struct.index(y)])
            for v in mods.values() if len(v) > 1 for i, x in enumerate(v) for y in v[i + 1:]]
    rows.append(dict(split=split, n1=int(h.sum()), n2=int((~h).sum()), corr_of_corr=r0, p_label_perm=(np.sum(np.array(nul) >= r0) + 1) / 1000,
                     module_pairs_same_sign=f"{int(np.sum(same))}/{len(same)}"))
REP = pd.DataFrame(rows); REP.to_csv(os.path.join(TAB, f"m6_module_reproducibility_{a.use}.tsv"), sep="\t", index=False)
for r in REP.itertuples():
    C.log(f"  module reproducibility ({r.split}): agreement of the halves' correlation matrices rho = {r.corr_of_corr:.2f} (label permutation p = {r.p_label_perm:.3g}); within-module pairs with the same sign {r.module_pairs_same_sign}")

# ------------------------------------------------------------------ 2. geometric coupling
FE = pd.concat([pd.DataFrame(M, columns=ML), MS], axis=1)
r, p, _ = C.block_perm_corr(FE.values, Gres, gx, gy, block=blk, n_perm=a.n_perm, seed=1, ranks=False)
GC = pd.DataFrame([dict(feature=f, geometry=g, rho=r[i, j], p_perm=p[i, j]) for i, f in enumerate(FE.columns) for j, g in enumerate(GEO.columns)])
GC["q"] = C.bh(GC.p_perm.values); GC["is_module"] = GC.feature.isin(MS.columns); GC["structured"] = GC.feature.isin(struct) | GC.is_module
GC = GC.sort_values("p_perm"); GC.to_csv(os.path.join(TAB, f"m6_geometry_coupling_{a.use}.tsv"), sep="\t", index=False)
sig = GC[(GC.q < 0.05) & GC.structured]
C.log(f"geometric coupling (block permutation, BH): {len(sig)} structured metabolite / module x geometry pairs with q < 0.05")
for x in sig.head(12).itertuples():
    C.log(f"   {x.feature:24s} ~ {x.geometry:22s} ρ = {x.rho:+.2f}  q = {x.q:.2g}")

# ------------------------------------------------------------------ 3. explanatory power
gfile = os.path.join(P["PROC"], f"m4_grid_IDHmut_{'' if a.ion_set == 'all' else a.ion_set + '_'}{R}um.parquet")
Gd = pd.read_parquet(gfile).set_index(["gx", "gy"]).reindex(pd.MultiIndex.from_arrays([gx, gy]))
gcols = [c for c in Gd if c.startswith("g:") and Gd[c].std() > 0]; ccols = [c for c in Gd if c.startswith("ct:")]
XT = Gd[ccols + gcols].fillna(0).values
bid = (gy // blk) * 10000 + (gx // blk); ub = np.unique(bid)
folds = np.array([dict(zip(ub, np.random.default_rng(1).permutation(np.arange(len(ub)) % 5)))[b] for b in bid])


def cvr2(Xf, y):
    yh = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        yh[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(Xf[~te], y[~te]).predict(Xf[te])
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


XG = np.column_stack([Gres, Gres ** 2])
from sklearn.decomposition import PCA
XT10 = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(XT))           # transcriptomic principal components (dimension comparable to geometry)
xs, ys = (gx - gx.mean()) / gx.std(), (gy - gy.mean()) / gy.std()
XC = np.column_stack([xs, ys, xs * xs, ys * ys, xs * ys])                                  # spatial-trend baseline (quadratic in coordinates)


def smooth_null_set():
    """Random smooth fields matched to the Moran's I of each geometric covariate (autocorrelation kept, biological meaning removed) -> null for the smoothness of geometry."""
    from scipy.ndimage import gaussian_filter
    out = []
    for j in range(Gres.shape[1]):
        im0 = np.full((gy.max() + 1, gx.max() + 1), np.nan); im0[gy, gx] = Gres[:, j]; I_t = C.morans_i_grid(im0)
        best = None
        for sg in (0.5, 1, 1.5, 2, 3, 4, 6, 8):
            w = gaussian_filter(rng.standard_normal((gy.max() + 1, gx.max() + 1)), sg)[gy, gx]
            im0[gy, gx] = w; I_ = C.morans_i_grid(im0)
            if best is None or abs(I_ - I_t) < best[0]:
                best = (abs(I_ - I_t), w)
        out.append(best[1])
    Zs = np.column_stack(out); return np.column_stack([Zs, Zs ** 2])


NULLSETS = [smooth_null_set() for _ in range(30)]
rows = []
for f in list(struct) + [k for k in MS.columns if not k.startswith("single:")]:
    y = FE[f].values
    oth = M[:, [i for i, m in enumerate(ML) if m != f and (f not in mods or m not in mods[f])]]
    nullr2 = [cvr2(Zn, y) for Zn in NULLSETS]
    rg = cvr2(XG, y)
    rows.append(dict(feature=f, is_module=f in MS.columns, R2_geometry=rg, R2_smooth_null_median=float(np.median(nullr2)),
                     R2_smooth_null_p95=float(np.percentile(nullr2, 95)), p_geometry_vs_smooth=(np.sum(np.array(nullr2) >= rg) + 1) / (len(nullr2) + 1),
                     R2_coordinates=cvr2(XC, y), R2_transcriptome_PC10=cvr2(XT10, y), R2_transcriptome=cvr2(XT, y),
                     R2_geometry_plus_transcriptome=cvr2(np.column_stack([XG, XT]), y), R2_other_metabolites=cvr2(oth, y)))
EV = pd.DataFrame(rows); EV.to_csv(os.path.join(TAB, f"m6_explained_variance_{a.use}.tsv"), sep="\t", index=False)
med = EV[["R2_geometry", "R2_smooth_null_median", "R2_coordinates", "R2_transcriptome", "R2_transcriptome_PC10", "R2_geometry_plus_transcriptome", "R2_other_metabolites"]].median()
EV["geometry_beats_smooth_null"] = EV.p_geometry_vs_smooth < 0.05
C.log(f"  features for which geometry beats smooth random fields of matched smoothness (p < 0.05): {int(EV.geometry_beats_smooth_null.sum())}/{len(EV)}")
EV.to_csv(os.path.join(TAB, f"m6_explained_variance_{a.use}.tsv"), sep="\t", index=False)
C.log("median block cross-validated R² (structured metabolites + modules): " + ", ".join(f"{k[3:]} {v:+.3f}" for k, v in med.items()))
nbetter = int(((EV.R2_geometry > EV.R2_transcriptome + 0.02) & EV.geometry_beats_smooth_null).sum())
C.log(f"  geometry > transcriptome (difference > 0.02) and geometry beats smooth random fields: {nbetter}/{len(EV)}")

# ------------------------------------------------------------------ figures
if len(S) > 1:
    o = leaves_list(linkage(squareform(1 - Cm, checks=False), "average")) if len(S) >= 3 else np.arange(len(S))
    fig, ax = plt.subplots(figsize=(.32 * len(S) + 2.5, .3 * len(S) + 1.5))
    im = ax.imshow(Cm[np.ix_(o, o)], cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar(im, ax=ax, shrink=.6, label="within-tumor residual r")
    ax.set_xticks(range(len(S))); ax.set_xticklabels([struct[i] for i in o], rotation=90); ax.set_yticks(range(len(S))); ax.set_yticklabels([struct[i] for i in o])
    ax.set_title("Metabolite modules within tumor (IDH-mut)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m6_module_heatmap_{a.use}.png"), dpi=160); plt.close(fig)
show = [k for k in MS.columns if not k.startswith("single:")][:4] or list(MS.columns[:4])
if show:
    fig, axs = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3), squeeze=False)
    for axx, k in zip(axs[0], show):
        im_ = np.full((gy.max() + 1, gx.max() + 1), np.nan); im_[gy, gx] = MS[k].values
        axx.imshow(im_, cmap="magma", interpolation="nearest"); axx.axis("off"); axx.set_title(f"{k}: {', '.join(mods[k])}"[:70], fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m6_module_maps_{a.use}.png"), dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(5, 3.4))
ev = EV.sort_values("R2_other_metabolites", ascending=False); xx = np.arange(len(ev)); w_ = .27
ax.bar(xx - w_, ev.R2_geometry.clip(lower=-.05), w_, label="tissue geometry", color="#31a354")
ax.scatter(xx - w_, ev.R2_smooth_null_p95.clip(lower=-.05), marker="_", s=40, c="k", zorder=3, label="95th pct, matched smooth random fields")
ax.bar(xx, ev.R2_transcriptome.clip(lower=-.05), w_, label="transcriptome (comp + genes)", color="#de2d26")
ax.bar(xx + w_, ev.R2_other_metabolites.clip(lower=-.05), w_, label="other metabolites (MALDI)", color="#636363")
ax.set_xticks(xx); ax.set_xticklabels(ev.feature, rotation=90); ax.axhline(0, c="k", lw=.4); ax.set_ylabel("spatial block-CV R² (within tumor)")
ax.legend(fontsize=6, frameon=False); ax.set_title("What organizes within-tumor metabolic structure?")
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m6_geometry_vs_transcriptome_{a.use}.png"), dpi=160); plt.close(fig)
C.save_json(dict(ion_set=a.ion_set, res_um=R, use=a.use, n_tumor_bins=int(len(gx)), structured=struct,
                 modules={k: v for k, v in mods.items() if len(v) > 1}, reproducibility=REP.to_dict(orient="records"),
                 geometry_q05=[f"{x.feature}~{x.geometry} ({x.rho:+.2f})" for x in sig.itertuples()],
                 median_R2=med.round(4).to_dict(), n_geometry_beats_transcriptome=nbetter, n_features=len(EV), runtime_s=round(time.time() - t0)),
            os.path.join(TAB, f"m6_02_summary_{a.use}.json"))
C.log(f"done ({time.time() - t0:.0f} s)")
