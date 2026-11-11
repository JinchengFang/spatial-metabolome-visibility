#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad, h5py
from scipy.ndimage import binary_closing, binary_fill_holes
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--sma-dir", default=None); ap.add_argument("--grid-um", type=float, default=50.0)
ap.add_argument("--scale-tol", type=float, default=0.03, help="with a pixel size in the imzML header, scale searched within +-scale-tol only (the raster step is a known physical quantity)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step10"); G = a.grid_um
D = a.sma_dir or os.path.join(P["PUB"], "Vicari2024_SMA_same_section"); OUT = os.path.join(P["PROC"], "sma"); plt.rcParams.update({"font.size": 7})


def read_visium(sec):
    """Section v11l12-038-a1 -> ST files V11L12-038_A1_*. Returns (barcodes, xy_um, in_tissue, counts csr, genes, region)."""
    slide, cap = sec.upper().replace("_", "-").rsplit("-", 1)
    slide = {"V11T17-085": "V11T16-085"}.get(slide, slide)          # MSI file name v11t17-085 corresponds to ST file name V11T16-085
    stem = f"{slide}_{cap}"
    def find(pat):
        g = sorted(glob.glob(os.path.join(D, "ST", f"{stem}*{pat}")) + glob.glob(os.path.join(D, "**", f"{stem}*{pat}"), recursive=True))
        return g[0] if g else None
    fh, fp, fs = find("filtered_feature_bc_matrix.h5"), find("tissue_positions*.csv"), find("scalefactors_json.json")
    if not (fh and fp and fs):
        return None
    with h5py.File(fh, "r") as f:
        m = f["matrix"]; shape = m["shape"][:]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape).T.tocsr()
        bc = [b.decode() for b in m["barcodes"][:]]; genes = [g.decode() for g in m["features/name"][:]]
    pos = pd.read_csv(fp, header=None if not open(fp).readline().startswith("barcode") else 0)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"][: pos.shape[1]]
    pos = pos.set_index("barcode"); u_ = C.visium_um_per_px(pos[["pxl_col", "pxl_row"]].values)   # calibrated on the 100-µm spot pitch
    xy_all = np.column_stack([pos.pxl_col.values, pos.pxl_row.values]) * u_        # all spots of the capture area (including off-tissue), for the capture-area mask
    pos = pos.reindex(bc)
    um_per_px = u_
    xy = np.column_stack([pos.pxl_col.values, pos.pxl_row.values]) * um_per_px
    reg = None; fr = find("region.csv")
    if fr:
        r_ = pd.read_csv(fr).set_index(pd.read_csv(fr).columns[0]); reg = r_.reindex(bc).iloc[:, 0].astype(str).values
    return dict(barcodes=np.array(bc), xy=xy, xy_all=xy_all, in_tissue=pos.in_tissue.values.astype(bool), X=X, genes=np.array(genes), region=reg, stem=stem)


def raster(xy, w, origin, shape, radius_px=0):
    M = np.zeros(shape, bool); ij = np.floor((xy - origin) / G).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0]); M[ij[ok, 1], ij[ok, 0]] = True
    if radius_px:
        from scipy.ndimage import binary_dilation
        yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]; M = binary_dilation(M, xx ** 2 + yy ** 2 <= radius_px ** 2)
    return binary_fill_holes(binary_closing(M, iterations=2))


def transform(xy, flip, theta, s, t=(0, 0)):
    q = xy.astype(float).copy()
    if flip:
        q[:, 0] = -q[:, 0]
    c, s_ = np.cos(theta), np.sin(theta); q = q @ np.array([[c, -s_], [s_, c]]).T * s
    return q + np.asarray(t)


def best_shift(A, B):
    """FFT cross-correlation: shift (dy, dx) of A with maximal overlap with B, and the overlap in pixels."""
    sh = (A.shape[0] + B.shape[0], A.shape[1] + B.shape[1])
    cc = np.fft.irfft2(np.fft.rfft2(B, sh) * np.conj(np.fft.rfft2(A, sh)), sh)
    k = np.unravel_index(np.argmax(cc), cc.shape); dy, dx = k
    if dy > sh[0] // 2:
        dy -= sh[0]
    if dx > sh[1] // 2:
        dx -= sh[1]
    return dy, dx, cc[k]


def best_shift_clip(A, B, F):
    """Shift maximizing IoU within the capture area F: two FFT cross-correlations give |A and B| and |A and F|; A must overlap the capture area with >= 30% of its tissue area."""
    sh = (A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]); fa = np.conj(np.fft.rfft2(A, sh))
    cb = np.fft.irfft2(np.fft.rfft2(B, sh) * fa, sh); cf = np.fft.irfft2(np.fft.rfft2(F, sh) * fa, sh)
    iou = np.where(cf > 0.3 * B.sum(), cb / np.maximum(cf + B.sum() - cb, 1e-9), -1)
    k = np.unravel_index(np.argmax(iou), iou.shape); dy, dx = k
    if dy > sh[0] // 2:
        dy -= sh[0]
    if dx > sh[1] // 2:
        dx -= sh[1]
    return dy, dx, cb[k], cf[k]


rows = []
for fm in sorted(glob.glob(os.path.join(OUT, "*_msi.h5ad"))):
    sec = os.path.basename(fm).replace("_msi.h5ad", ""); t0 = time.time()
    V = read_visium(sec)
    if V is None:
        C.log(f"[{sec}] no matching Visium files; skipped"); continue
    M = ad.read_h5ad(fm); xy_m = np.column_stack([M.obs.x.values, M.obs.y.values]).astype(float)
    vt = V["xy"][V["in_tissue"]]; area_um2 = V["in_tissue"].sum() * (100.0 ** 2 * np.sqrt(3) / 2)
    # tissue signal: summed raw intensity at m/z 600-1000 (membrane lipids); matrix (DHB / 9-AA) cluster ions are concentrated at low mass and stronger off tissue,
    # so TIC or the strongest peaks cannot be used. Falls back to all peaks if there are too few
    from scipy.ndimage import binary_fill_holes as _fh, label as _lab
    mzv = M.var.mz.values; lip = np.where((mzv > 600) & (mzv < 1000))[0]
    lip = lip if len(lip) >= 20 else np.arange(M.n_vars)
    lt = np.log1p(np.asarray(M.X.tocsc()[:, lip].sum(1)).ravel())
    s_hdr = M.uns.get("pixel_size_um", np.nan); has_hdr = bool(s_hdr == s_hdr and s_hdr)
    def make_mask(thr):
        img = np.zeros((int(M.obs.y.max()) + 2, int(M.obs.x.max()) + 2), bool); img[M.obs.y.values.astype(int), M.obs.x.values.astype(int)] = lt > thr
        img = _fh(img); lb, nl = _lab(img)
        if nl > 1:                                 # keep the largest connected tissue region
            sizes = np.bincount(lb.ravel()); sizes[0] = 0; img = lb == np.argmax(sizes)
        return img[M.obs.y.values.astype(int), M.obs.x.values.astype(int)]
    # candidate thresholds: Otsu and quantiles matched to the Visium tissue area (with known pixel size)
    thrs = {"all": -np.inf, "otsu": C.otsu(lt[lt > 0])[0]}       # all = every pixel of the acquisition area (SMA MALDI acquisition areas follow the tissue outline)
    if has_hdr:
        f_target = min(0.98, area_um2 / (float(s_hdr) ** 2) / len(lt))
        for k_ in (0.9, 1.0, 1.1, 1.2):
            f_ = min(0.98, f_target * k_); thrs[f"area×{k_}"] = float(np.quantile(lt, 1 - f_))
    o_v = vt.min(0) - 500; shp = (int((vt[:, 1].max() - o_v[1] + 500) / G) + 1, int((vt[:, 0].max() - o_v[0] + 500) / G) + 1)
    B = raster(vt, None, o_v, shp, radius_px=int(60 / G)).astype(float)
    # Visium capture area (all spots, including off-tissue): the tissue is often larger than the 6.5-mm capture area (MALDI covers the whole tissue) ->
    # IoU computed within the capture area: |A and B| / (|A and F| + |B| - |A and B|), F = capture area
    Fm = np.maximum(raster(V["xy_all"], None, o_v, shp, radius_px=int(60 / G)).astype(float), B)

    def scorer(pm):
        def score(flip, th, s):
            q = transform(pm, flip, th, s) + vt.mean(0)
            o = q.min(0) - 500; sh_ = (int((q[:, 1].max() - o[1] + 500) / G) + 1, int((q[:, 0].max() - o[0] + 500) / G) + 1)
            A = raster(q, None, o, sh_, radius_px=max(1, int(0.6 * s / G))).astype(float)
            dy, dx, ov, af = best_shift_clip(A, B, Fm)
            t = (o_v - o) + np.array([dx, dy]) * G             # translation (µm): origin o of A moved to origin o_v of B + grid shift
            return ov / (af + B.sum() - ov), t
        return score

    def search(score, s0, rng_s, step_th=3.0):
        cands = {}
        for flip in (False, True):
            best = (-1,)
            for th in np.deg2rad(np.arange(0, 360, step_th)):
                for s in s0 * rng_s:
                    iou, t = score(flip, th, s)
                    if iou > best[0]:
                        best = (iou, flip, th, s, t)
            cands[flip] = best
        return cands
    # round 1: coarse search (6-degree rotation steps) for every candidate mask; the mask with the highest IoU is kept
    mask_rows = []; bestmask = None
    for name, thr in thrs.items():
        tis_ = make_mask(thr)
        if tis_.sum() < 50:
            continue
        s0_ = float(s_hdr) if has_hdr else float(np.sqrt(area_um2 / tis_.sum()))
        rng_s = np.linspace(1 - a.scale_tol, 1 + a.scale_tol, 3) if has_hdr else np.arange(0.85, 1.151, 0.05)
        cs = search(scorer(xy_m[tis_] - xy_m[tis_].mean(0)), s0_, rng_s, step_th=6.0)
        b = max(cs.values(), key=lambda z: z[0]); mask_rows.append(dict(mask=name, n_tissue_px=int(tis_.sum()), iou_coarse=b[0]))
        if bestmask is None or b[0] > bestmask[0]:
            bestmask = (b[0], name, tis_, s0_, rng_s)
    _, mask_name, tis, s0, rng_s = bestmask
    C.log(f"[{sec}] tissue-mask candidates: " + "; ".join(f"{r['mask']} {r['n_tissue_px']} pixels IoU {r['iou_coarse']:.2f}" for r in mask_rows) + f" -> {mask_name}")
    pm = xy_m[tis] - xy_m[tis].mean(0); score = scorer(pm)
    # round 2: fine search on the selected mask (3-degree steps) + local refinement (0.5 degrees, 1%)
    cands = search(score, s0, rng_s, step_th=3.0)
    for flip in (False, True):
        best = cands[flip]; _, fl, th0, s0b, _ = best
        for th in th0 + np.deg2rad(np.arange(-3, 3.01, 0.5)):
            for s in s0b * np.arange(0.97, 1.031, 0.01):
                if has_hdr and abs(s / s0 - 1) > a.scale_tol + 1e-9:
                    continue
                iou, t = score(fl, th, s)
                if iou > best[0]:
                    best = (iou, fl, th, s, t)
        cands[flip] = best
    # diagnostic (spot-level maximum |r| between metabolome and transcriptome PCs)
    from scipy.spatial import cKDTree
    def diag(q):
        tree = cKDTree(q[tis]); nb = tree.query_ball_point(vt, r=55)
        Xm = M.X.tocsr()[np.where(tis)[0]]; Xm = sp.diags(1 / np.maximum(M.obs.tic.values[tis], 1)) @ Xm
        keep = [i for i, n_ in enumerate(nb) if len(n_) >= 1]
        if len(keep) < 50:
            return np.nan
        Ym = np.log1p(np.vstack([np.asarray(Xm[nb[i]].mean(0)).ravel() for i in keep]) * 1e4)
        Xv = V["X"][np.where(V["in_tissue"])[0]][keep]; Xv = np.log1p((sp.diags(1e4 / np.maximum(np.asarray(Xv.sum(1)).ravel(), 1)) @ Xv).toarray())
        Xv = Xv[:, Xv.var(0) > 0]; hv = np.argsort(-Xv.var(0))[:2000]
        pc = lambda Z: np.linalg.svd(C.zcols(Z), full_matrices=False)[0][:, :3]
        R_ = np.corrcoef(pc(Ym).T, pc(Xv[:, hv]).T)[:3, 3:]
        return float(np.abs(R_).max())
    qmap = lambda b, extra=0.0: transform(xy_m - xy_m[tis].mean(0), b[1], b[2] + extra, b[3]) + vt.mean(0) + b[4]
    dg = {f_: diag(qmap(b)) for f_, b in cands.items()}
    # choice: IoU higher by > 0.03 decides; otherwise (e.g. symmetric coronal sections) the molecular diagnostic breaks the tie, as noted in the table
    i0, i1 = cands[False][0], cands[True][0]
    if abs(i0 - i1) > 0.03:
        pick = i1 > i0; how = "IoU"
    else:
        pick = bool(np.nan_to_num(dg[True]) > np.nan_to_num(dg[False]) + 0.02); how = "symmetric outline; flip decided by molecular check"
    iou, fl, th, s, t = cands[pick]; q_all = qmap(cands[pick])
    d_best = dg[pick]; d_rot = diag(qmap(cands[pick], np.pi))
    M.obs["x_vis_um"], M.obs["y_vis_um"] = q_all[:, 0], q_all[:, 1]; M.obs["tissue"] = tis
    M.uns.update(dict(reg_iou=float(iou), reg_flip=bool(fl), reg_theta_deg=float(np.rad2deg(th) % 360), reg_scale_um_per_px=float(s), visium_stem=V["stem"]))
    M.write_h5ad(os.path.join(OUT, f"{sec}_paired.h5ad"), compression="gzip")
    rows.append(dict(section=sec, visium=V["stem"], n_pixels=M.n_obs, tissue_pixels=int(tis.sum()), tissue_mask=mask_name, spots_in_tissue=int(V["in_tissue"].sum()), iou=iou, flip=fl,
                     iou_noflip=i0, iou_flip=i1, diag_noflip=dg[False], diag_flip=dg[True], flip_decided_by=how,
                     rotation_deg=np.rad2deg(th) % 360, um_per_pixel=s, um_per_pixel_init=s0, diag_pc_r_best=d_best, diag_pc_r_rot180=d_rot, seconds=time.time() - t0))
    C.log(f"[{sec} - {V['stem']}] IoU {iou:.3f}, flip {fl}, rotation {np.rad2deg(th) % 360:.1f} deg, {s:.1f} µm/pixel (start {s0:.1f}); "
          f"diagnostic |r|(PC) best {d_best:.2f} vs 180-degree rotation {d_rot:.2f}" + ("  WARNING: IoU < 0.8, check the registration plot" if iou < 0.8 else ""))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(vt[:, 0], vt[:, 1], s=3, c="#bdbdbd", label="Visium in-tissue spots")
    sel = np.where(tis)[0][:: max(1, tis.sum() // 6000)]
    ax.scatter(q_all[sel, 0], q_all[sel, 1], s=1, c=np.log1p(M.obs.tic.values[sel]), cmap="viridis", alpha=.6, label="MALDI pixels (log TIC)")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(f"{sec} ↔ {V['stem']}  IoU {iou:.2f}", fontsize=7); ax.legend(fontsize=5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m13_registration_{sec}.png"), dpi=150); plt.close(fig)
if rows:
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "m13_registration.tsv"), sep="\t", index=False)
    C.log("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
else:
    C.log("no sections to register (run M13-01 first)")
