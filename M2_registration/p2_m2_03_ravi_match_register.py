#!/usr/bin/env python3

import argparse, glob, os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.signal import fftconvolve
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--candidates", default="UKF275,UKF262,UKF260,UKF259,UKF248,UKF296")
ap.add_argument("--n-perm", type=int, default=999)
ap.add_argument("--map-radius", type=float, default=50.0)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args(); P = C.paths(a.root)
TAB = os.path.join(os.path.dirname(P["TAB"]), "Step2"); FIG = os.path.join(os.path.dirname(P["FIG"]), "Step2")
OUT = os.path.join(P["PROC"], "ravi_paired")
for d in (TAB, FIG, OUT):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 7})

secs = {os.path.basename(f)[:-5]: f for f in sorted(glob.glob(os.path.join(P["PROC"], "ravi_maldi_sections", "S*.h5ad")))}
vis = {os.path.basename(f)[:-5]: f for f in sorted(glob.glob(os.path.join(P["PROC"], "ravi_visium", "*.h5ad")))}
if not secs or not vis:
    raise SystemExit("ravi_maldi_sections or ravi_visium missing; run run_step2.sh 1-2 first")
cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
cand_vis = {}
for c in cands:
    hit = [s for s in vis if s.startswith(c + "_T") and "IDH" not in s] or [s for s in vis if s.startswith(c)]
    if hit:
        cand_vis[c] = sorted(hit, key=len)[0]
C.log(f"{len(secs)} MALDI sections; {len(vis)} Visium samples; listed candidates -> {cand_vis}")


# ----------------------------------------------------------------------------- geometry
def raster_pts(pts, res, radius, origin=None, shape=None):
    o = pts.min(0) - 2 * radius if origin is None else origin
    ij = np.floor((pts - o) / res).astype(int)
    if shape is None:
        shape = (ij[:, 1].max() + 1 + int(2 * radius / res) + 1, ij[:, 0].max() + 1 + int(2 * radius / res) + 1)
    M = np.zeros(shape, bool)
    r = int(np.ceil(radius / res))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx * dx + dy * dy) * res * res <= radius * radius + 1e-9:
                yy, xx = ij[:, 1] + dy, ij[:, 0] + dx
                ok = (yy >= 0) & (xx >= 0) & (yy < shape[0]) & (xx < shape[1])
                M[yy[ok], xx[ok]] = True
    return M, o


def transform(pts, flip, theta, s, c_src, t):
    p = pts.copy()
    if flip:
        p[:, 0] = 2 * c_src[0] - p[:, 0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return s * (p - c_src) @ R.T + t


def best_shift(D, Dor, S, Sor, res):
    """FFT cross-correlation: translation of S onto D with maximal overlap. Returns IoU and translation (µm)."""
    cc = fftconvolve(D.astype(float), S[::-1, ::-1].astype(float), mode="full")
    i, j = np.unravel_index(np.argmax(cc), cc.shape)
    ov = cc[i, j]
    iou = ov / (D.sum() + S.sum() - ov + 1e-9)
    new_or = Dor + np.array([(j - (S.shape[1] - 1)) * res, (i - (S.shape[0] - 1)) * res])
    return float(iou), new_or - Sor


def register(src, dst, src_r=25.0, dst_r=60.0):
    """Similarity-transform search (src -> dst). Returns (IoU, parameters); src and dst are point sets in µm."""
    c_src = src.mean(0)
    best = (-1, None)
    for res, thetas, scales, flips in (
            (100.0, np.deg2rad(np.arange(0, 360, 6)), np.arange(0.85, 1.151, 0.05), (False, True)),
            (50.0, None, None, None)):
        D, Dor = raster_pts(dst, res, max(dst_r, res / 2))
        if thetas is None:
            _, pr = best
            thetas = pr["theta"] + np.deg2rad(np.arange(-6, 6.1, 1)); scales = pr["s"] + np.arange(-0.05, 0.051, 0.01); flips = (pr["flip"],)
            best = (-1, None)
        for fl in flips:
            for th in thetas:
                for s in scales:
                    q = transform(src, fl, th, s, c_src, np.zeros(2))
                    S, Sor = raster_pts(q, res, max(src_r, res / 2))
                    iou, sh = best_shift(D, Dor, S, Sor, res)
                    if iou > best[0]:
                        best = (iou, dict(flip=bool(fl), theta=float(th), s=float(s), c_src=c_src.tolist(), t=sh.tolist()))
    return best


def reg_job(args):
    k, src, v, dst = args
    iou, pr = register(src, dst)
    return k, v, iou, pr


# ----------------------------------------------------------------------------- input
S_pts = {}
for k, f in secs.items():
    A = ad.read_h5ad(f, backed="r"); S_pts[k] = A.obs[["x_local_um", "y_local_um"]].values.astype(float); A.file.close()
V_pts, V_meta = {}, {}
for k, f in vis.items():
    B = ad.read_h5ad(f, backed="r"); V_pts[k] = B.obsm["spatial_um"][:].astype(float); V_meta[k] = dict(B.uns); B.file.close()

# ----------------------------------------------------------------------------- 1a. shape: all sections x all Visium samples
from concurrent.futures import ProcessPoolExecutor
jobs = [(k, S_pts[k], v, V_pts[v]) for k in secs for v in vis]
C.log(f"shape registration search: {len(jobs)} (section, Visium sample) pairs, {a.workers} processes ...")
res_all = {}
with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
    for k, v, iou, pr in ex.map(reg_job, jobs, chunksize=2):
        res_all[(k, v)] = (iou, pr)
IOU = pd.DataFrame({v: {k: res_all[(k, v)][0] for k in secs} for v in vis}).loc[list(secs)]
IOU.to_csv(os.path.join(TAB, "ravi_identity_shape_iou_all.tsv"), sep="\t")
cv = [cand_vis[c] for c in cands if c in cand_vis]
Ic = IOU[cv]
ri, ci = linear_sum_assignment(-Ic.values)
assign = {Ic.index[i]: Ic.columns[j] for i, j in zip(ri, ci)}
rows = []
for k, v in assign.items():
    rank_all = int((IOU.loc[k] > IOU.loc[k, v]).sum()) + 1
    second = Ic.loc[k].drop(v).max()
    rows.append(dict(section=k, visium=v, patient=V_meta[v].get("patient", v), iou=IOU.loc[k, v], rank_among_all_visium=rank_all,
                     n_visium=IOU.shape[1], second_best_candidate_iou=second, margin=IOU.loc[k, v] - second))
    C.log(f"  {k} -> {v}: IoU {IOU.loc[k, v]:.3f} (rank {rank_all} of {IOU.shape[1]} Visium samples; margin over the next candidate {IOU.loc[k, v] - second:+.3f})")
ID = pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 1b. molecular: 6 x 6 positive controls
def tic_norm(X):
    t = X.sum(1); return X / np.maximum(t, 1e-12)[:, None] * np.median(t[t > 0])


def peak_cols(var, names, tiers=("A", "B")):
    m = var["best_name"].astype(str).isin(names) & var["tier"].astype(str).isin(tiers)
    return np.where(m.values)[0]


SECDATA = {}
for k, f in secs.items():
    A = ad.read_h5ad(f)
    X = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
    SECDATA[k] = dict(X=np.log1p(tic_norm(X)).astype(np.float32), var=A.var.copy(), pts=S_pts[k], obs=A.obs.copy(), uns=dict(A.uns))
var0 = next(iter(SECDATA.values()))["var"]
if "tier" not in var0:
    raise SystemExit("MALDI sections have no annotation columns (tier / best_name); run run_step1.sh 4 and run_step2.sh 1 first")
SULF = [n for n in var0.best_name.astype(str).unique() if n.startswith("ST ") or n.startswith("HexCer")]
CTRL = {"sulfatide/HexCer ~ WM_oligodendrocyte": (SULF, "prog_WM_oligodendrocyte"),
        "N-acetylaspartate ~ Neuron_cortex": (["N-acetylaspartate"], "prog_Neuron_cortex")}
C.log("positive-control features: " + "; ".join(f"{k}: {len(peak_cols(var0, v[0]))} peaks" for k, v in CTRL.items()))
VDATA = {v: ad.read_h5ad(vis[v]) for v in set(cv)}


def map_to_spots(src_pts, pr, spots_um, radius):
    q = transform(src_pts, pr["flip"], pr["theta"], pr["s"], np.array(pr["c_src"]), np.array(pr["t"]))
    tree = cKDTree(q)
    nb = tree.query_ball_point(spots_um, r=radius)
    rows_, cols_ = [], []
    for i, js in enumerate(nb):
        rows_ += [i] * len(js); cols_ += js
    W = sp.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(len(spots_um), len(q)))
    n = np.asarray(W.sum(1)).ravel()
    W = sp.diags(1 / np.maximum(n, 1)) @ W
    return q, W, n


MOL = {name: pd.DataFrame(index=list(secs), columns=cv, dtype=float) for name in CTRL}
for k in secs:
    for v in cv:
        iou, pr = res_all[(k, v)]
        B = VDATA[v]; q, W, n = map_to_spots(SECDATA[k]["pts"], pr, B.obsm["spatial_um"], a.map_radius)
        ok = n > 0
        for name, (mets, prog) in CTRL.items():
            js = peak_cols(SECDATA[k]["var"], mets)
            if len(js) == 0 or prog not in B.obs or ok.sum() < 30:
                continue
            m = np.asarray(W[ok] @ SECDATA[k]["X"][:, js]).mean(1)
            MOL[name].loc[k, v] = spearmanr(m, B.obs[prog].values[ok])[0]
molsum = sum(M_.fillna(0) for M_ in MOL.values()) / max(1, len(MOL))
for name, M_ in MOL.items():
    M_.to_csv(os.path.join(TAB, f"ravi_identity_molecular_{name.split(' ~ ')[0].replace('/', '_')}.tsv"), sep="\t")
ID["molecular_score"] = [molsum.loc[r.section, r.visium] for r in ID.itertuples()]
ID["molecular_rank_in_row"] = [int((molsum.loc[r.section] > molsum.loc[r.section, r.visium]).sum()) + 1 for r in ID.itertuples()]
ID["identity_confidence"] = np.where((ID.rank_among_all_visium == 1) & (ID.molecular_rank_in_row == 1), "high (shape and molecular rank 1)",
                              np.where((ID.rank_among_all_visium <= 3) | (ID.molecular_rank_in_row == 1), "medium", "low (H&E check needed)"))
ID.to_csv(os.path.join(TAB, "ravi_identity_assignment.tsv"), sep="\t", index=False)
for r in ID.itertuples():
    C.log(f"  {r.section} -> {r.patient}: molecular control score {r.molecular_score:+.3f} (rank {r.molecular_rank_in_row} in row); confidence {r.identity_confidence}")

fig, axs = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw=dict(width_ratios=[2.6, 1, 1]))
im = axs[0].imshow(IOU.values, cmap="viridis", aspect="auto"); axs[0].set_yticks(range(len(IOU))); axs[0].set_yticklabels(IOU.index)
axs[0].set_xticks(range(IOU.shape[1])); axs[0].set_xticklabels(IOU.columns, rotation=90); fig.colorbar(im, ax=axs[0], label="shape IoU")
for r in ID.itertuples():
    axs[0].add_patch(plt.Rectangle((list(IOU.columns).index(r.visium) - .5, list(IOU.index).index(r.section) - .5), 1, 1, fill=False, ec="r", lw=2))
axs[0].set_title("blind shape matching: MALDI sections × all Visium samples (red = assignment)")
for axx, M_, t in ((axs[1], Ic, "shape IoU (README candidates)"), (axs[2], molsum.astype(float), "molecular control score")):
    im = axx.imshow(M_.values.astype(float), cmap="magma", aspect="auto"); fig.colorbar(im, ax=axx)
    axx.set_yticks(range(len(M_))); axx.set_yticklabels(M_.index); axx.set_xticks(range(M_.shape[1])); axx.set_xticklabels(M_.columns, rotation=90); axx.set_title(t)
    for r in ID.itertuples():
        axx.add_patch(plt.Rectangle((list(M_.columns).index(r.visium) - .5, list(M_.index).index(r.section) - .5), 1, 1, fill=False, ec="c", lw=2))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ravi_identity_matrices.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------- 2-4. registration, mapping, effective resolution
specs, ctrl_rows, reg_rows = [], [], []
for r in ID.itertuples():
    k, v, pid = r.section, r.visium, r.patient
    iou, pr = res_all[(k, v)]
    B = VDATA[v].copy(); D_ = SECDATA[k]
    q, W, n = map_to_spots(D_["pts"], pr, B.obsm["spatial_um"], a.map_radius)
    ok = n > 0
    Araw = ad.read_h5ad(secs[k]); Xraw = Araw.X.toarray() if sp.issparse(Araw.X) else np.asarray(Araw.X)
    B.obsm["metab"] = np.asarray(W @ tic_norm(Xraw)).astype(np.float32)
    B.obsm["maldi_px_map"] = W.tocsr()
    B.obs["n_maldi_px"] = n; B.obs["has_maldi"] = ok
    mv = D_["var"].copy(); mv.index = mv.index.astype(str)
    B.uns["metab_var_names"] = np.asarray(mv.index); B.uns["maldi_section"] = k
    B.uns["transform"] = {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in pr.items()}
    B.uns["registration_iou"] = float(iou)
    B.write_h5ad(os.path.join(OUT, f"{pid}_visium.h5ad"), compression="gzip")
    mv.to_csv(os.path.join(OUT, f"{pid}_metab_var.tsv"), sep="\t")
    Araw.obs["x_vis_um"] = q[:, 0]; Araw.obs["y_vis_um"] = q[:, 1]; Araw.obsm["spatial_visium_um"] = q
    Araw.uns["patient"] = pid; Araw.uns["visium_sample"] = v
    Araw.write_h5ad(os.path.join(OUT, f"{pid}_maldi.h5ad"), compression="gzip")
    reg_rows.append(dict(patient=pid, section=k, visium=v, iou=iou, flip=pr["flip"], rotation_deg=np.rad2deg(pr["theta"]) % 360, scale=pr["s"],
                         spots_in_tissue=B.n_obs, spots_with_maldi=int(ok.sum()), frac_spots_with_maldi=ok.mean(), median_px_per_spot=float(np.median(n[ok])) if ok.any() else 0))
    # ---- positive controls (spot level, block permutation)
    ix = B.obs.array_col.values // 2; iy = B.obs.array_row.values
    for name, (mets, prog) in CTRL.items():
        js = peak_cols(D_["var"], mets)
        if len(js) == 0 or ok.sum() < 30:
            continue
        m = np.log1p(B.obsm["metab"][ok][:, js]).mean(1)
        rr_, pp, _ = C.block_perm_test(m, B.obs[prog].values[ok], ix[ok], iy[ok], block=5, n_perm=a.n_perm)
        ctrl_rows.append(dict(patient=pid, pair=name, n_spots=int(ok.sum()), rho=float(rr_[0]), p_block=float(pp[0]),
                              p_naive=float(spearmanr(m, B.obs[prog].values[ok]).pvalue)))
    # ---- scale spectrum (50-µm raster, Visium coordinates; metabolites from raw pixels, not spot means)
    res = 50.0; o = np.minimum(q.min(0), B.obsm["spatial_um"].min(0)) - 200
    shp = (int((max(q[:, 1].max(), B.obsm["spatial_um"][:, 1].max()) - o[1]) / res) + 6, int((max(q[:, 0].max(), B.obsm["spatial_um"][:, 0].max()) - o[0]) / res) + 6)
    def mimg(js):
        return C.rasterize(q[:, 0], q[:, 1], D_["X"][:, js].mean(1), res, o[0], o[1], shp)[0]
    def gimg(col):
        return C.rasterize(B.obsm["spatial_um"][:, 0], B.obsm["spatial_um"][:, 1], B.obs[col].values, res, o[0], o[1], shp, radius_um=60)[0]
    pairs = {}
    for name, (mets, prog) in CTRL.items():
        js = peak_cols(D_["var"], mets)
        if len(js):
            pairs[name] = (mimg(js), gimg(prog))
    ceil = {}
    var_ = D_["var"]
    for nm in var_.best_name[var_.tier.astype(str) == "A"].astype(str).unique():
        js = peak_cols(var_, [nm], ("A",))
        if len(js) >= 2:
            js = js[np.argsort(-var_.frac_pixels.values[js])][:2]
            ceil[nm] = (mimg([js[0]]), mimg([js[1]]))
    bands = [b for b in C.BANDS if b[0] >= 100]
    sp1 = C.scale_spectrum(pairs, res, bands, "cross-modal"); sp2 = C.scale_spectrum(ceil, res, bands, "MALDI ceiling")
    s_ = pd.concat([sp1, sp2]); s_["patient"] = pid; specs.append(s_)
    # ---- overlay
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(B.uns["hires_image"]); f_ = B.uns["scalefactors"]["tissue_hires_scalef"] / B.uns["um_per_px"]
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(img); axs[0].scatter(q[::5, 0] * f_, q[::5, 1] * f_, s=0.3, c="c", alpha=.25); axs[0].set_title(f"{pid}: H&E + MALDI {k} (IoU {iou:.2f})")
        for axx, (t, im_) in zip(axs[1:], [(f"{nm}: MALDI", pairs[nm][0]) for nm in pairs] + [(f"{CTRL[nm][1]}: Visium", pairs[nm][1]) for nm in list(pairs)[:1]]):
            lo, hi = np.nanpercentile(im_, [2, 98]) if np.isfinite(im_).any() else (0, 1); axx.imshow(im_, cmap="viridis", vmin=lo, vmax=hi); axx.set_title(t, fontsize=6)
        for axx in axs:
            axx.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"ravi_{pid}_overlay.png"), dpi=140); plt.close(fig)
    except Exception as e:
        C.log(f"  [note] overlay for {pid} failed: {e}")
REG = pd.DataFrame(reg_rows); REG.to_csv(os.path.join(TAB, "ravi_registration.tsv"), sep="\t", index=False)
CT = pd.DataFrame(ctrl_rows); CT.to_csv(os.path.join(TAB, "ravi_controls.tsv"), sep="\t", index=False)
SP = pd.concat(specs) if specs else pd.DataFrame(); SP.to_csv(os.path.join(TAB, "ravi_scale_spectrum.tsv"), sep="\t", index=False)
for r in CT.itertuples():
    C.log(f"  control {r.patient} {r.pair}: rho = {r.rho:+.3f} (p_block {r.p_block:.3g}; parametric p {r.p_naive:.1e})")
if len(SP):
    med = SP.groupby(["band_lo", "band_um", "kind"]).r.median().unstack("kind").reset_index().sort_values("band_lo")
    for _, r_ in med.iterrows():
        C.log(f"  band {r_['band_um']:>9} µm: median cross-modal r = {r_.get('cross-modal', np.nan):.2f}   MALDI ceiling = {r_.get('MALDI ceiling', np.nan):.2f}")
    cm = med.set_index("band_lo")["cross-modal"] if "cross-modal" in med else pd.Series(dtype=float)
    eff = next((med.set_index("band_lo").loc[lo, "band_um"] for lo, v in cm.items() if v >= 0.3), None)
    C.save_json(dict(effective_band_um=eff, median_by_band=med.to_dict(orient="records"),
                     n_patients=int(REG.shape[0]), identity=ID.to_dict(orient="records")), os.path.join(TAB, "ravi_registration_summary.json"))
    C.log(f"  -> GBM effective resolution: {eff} µm")
    fig, axx = plt.subplots(figsize=(5, 3.4))
    for (kind, pid), d in SP.groupby(["kind", "patient"]):
        d = d.groupby("band_lo").r.median(); axx.plot(d.index, d.values, "-", color="#c0392b" if kind == "cross-modal" else "#2c7fb8", alpha=.35)
    for kind, col in (("cross-modal", "#c0392b"), ("MALDI ceiling", "#2c7fb8")):
        d = SP[SP.kind == kind].groupby("band_lo").r.median(); axx.plot(d.index, d.values, "o-", color=col, lw=2.5, label=kind)
    axx.set_xscale("log"); axx.axhline(.3, ls="--", c="grey"); axx.set_xlabel("band lower edge (µm)"); axx.set_ylabel("r"); axx.legend()
    axx.set_title("GBM (Ravi, 6 patients): scale-resolved MALDI↔Visium concordance"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ravi_scale_spectrum.png"), dpi=150); plt.close(fig)
C.log(f"→ {OUT}")
