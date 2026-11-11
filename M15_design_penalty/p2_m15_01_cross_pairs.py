#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, h5py, anndata as ad
from scipy.spatial import cKDTree
from scipy.ndimage import binary_closing, binary_fill_holes
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--grid-um", type=float, default=50.0)
ap.add_argument("--n-hvg", type=int, default=2000); ap.add_argument("--n-pc", type=int, default=10); ap.add_argument("--sma-dir", default=None)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step11"); G = a.grid_um; t00 = time.time()
D = a.sma_dir or os.path.join(P["PUB"], "Vicari2024_SMA_same_section")
OUT = os.path.join(P["PROC"], "sma_cross"); os.makedirs(OUT, exist_ok=True)


def read_vis(stem):
    f = lambda pat: (sorted(glob.glob(os.path.join(D, "**", f"{stem}*{pat}"), recursive=True)) or [None])[0]
    fh, fp = f("filtered_feature_bc_matrix.h5"), f("tissue_positions*.csv")
    with h5py.File(fh, "r") as h:
        m = h["matrix"]; X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=m["shape"][:]).T.tocsr()
        bc = [b.decode() for b in m["barcodes"][:]]; genes = np.array([g.decode().upper() for g in m["features/name"][:]])
    pos = pd.read_csv(fp, header=None if not open(fp).readline().startswith("barcode") else 0)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"][: pos.shape[1]]; pos = pos.set_index("barcode")
    um = C.visium_um_per_px(pos[["pxl_col", "pxl_row"]].values); pos = pos.reindex(bc)
    _, gi = np.unique(genes, return_index=True); gi = np.sort(gi); genes = genes[gi]; X = X[:, gi]
    keep = pos.in_tissue.values.astype(bool) & (np.asarray(X.sum(1)).ravel() > 0)
    X = X[np.where(keep)[0]]; tot = np.asarray(X.sum(1)).ravel()
    LG = (sp.diags(1e4 / np.maximum(tot, 1)) @ X).tocsr(); LG.data = np.log1p(LG.data)
    return dict(stem=stem, xy=np.column_stack([pos.pxl_col.values, pos.pxl_row.values])[keep] * um, LG=LG, genes=genes)


def shared_pcs(A, B):
    g = np.intersect1d(A["genes"], B["genes"]); ma = {x: i for i, x in enumerate(A["genes"])}; mb = {x: i for i, x in enumerate(B["genes"])}
    ia = np.array([ma[x] for x in g]); ib = np.array([mb[x] for x in g])
    XA, XB = A["LG"][:, ia].toarray(), B["LG"][:, ib].toarray()
    v = XA.var(0) + XB.var(0); hv = np.argsort(-v)[: a.n_hvg]
    ZA, ZB = C.zcols(XA[:, hv]), C.zcols(XB[:, hv]); Z = np.vstack([ZA, ZB])
    U, S_, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); W = Vt[: a.n_pc].T
    return C.zcols(ZA @ W), C.zcols(ZB @ W)


def raster(xy, origin, shape):
    M = np.zeros(shape, bool); ij = np.floor((xy - origin) / G).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0]); M[ij[ok, 1], ij[ok, 0]] = True
    return binary_fill_holes(binary_closing(M, iterations=2)).astype(float)


def rot(q, flip, th):
    q = q.copy()
    if flip:
        q[:, 0] = -q[:, 0]
    c, s = np.cos(th), np.sin(th)
    return q @ np.array([[c, -s], [s, c]]).T


def agreement(Q, PA, treeB, PB):
    d, j = treeB.query(Q); ok = d < 75
    if ok.sum() < 50:
        return -1.0, 0.0
    pa, pb = PA[ok], PB[j[ok]]
    r = [np.corrcoef(pa[:, k], pb[:, k])[0, 1] for k in range(pa.shape[1])]
    return float(np.nanmean(r)), float(ok.mean())


def register_vv(A, B, PA, PB):
    """Rigid transform Visium A -> Visium B. Returns (score, flip, theta, t, overlap, score_other_flip)."""
    ca = A["xy"].mean(0); qa0 = A["xy"] - ca; treeB = cKDTree(B["xy"])
    oB = B["xy"].min(0) - 500; shB = tuple((np.ceil((B["xy"].max(0) + 500 - oB) / G).astype(int))[::-1]); MB = raster(B["xy"], oB, shB)
    def coarse_t(Q):
        oQ = Q.min(0) - 500; shQ = tuple((np.ceil((Q.max(0) + 500 - oQ) / G).astype(int))[::-1]); MQ = raster(Q, oQ, shQ)
        sh = (MQ.shape[0] + MB.shape[0], MQ.shape[1] + MB.shape[1])
        cc = np.fft.irfft2(np.fft.rfft2(MB, sh) * np.conj(np.fft.rfft2(MQ, sh)), sh); k = np.unravel_index(np.argmax(cc), cc.shape)
        dy, dx = [k[i] if k[i] <= sh[i] // 2 else k[i] - sh[i] for i in range(2)]
        return np.array([dx, dy]) * G + oB - oQ
    best = {}
    for flip in (False, True):
        bb = (-2, None)
        for th in np.deg2rad(np.arange(0, 360, 3.0)):
            Q = rot(qa0, flip, th); t = coarse_t(Q); s_, ov = agreement(Q + t, PA, treeB, PB)
            if s_ > bb[0]:
                bb = (s_, th, t, ov)
        s0, th0, t0, ov0 = bb
        for th in th0 + np.deg2rad(np.arange(-3, 3.01, 0.5)):        # refine rotation and translation
            Q = rot(qa0, flip, th)
            for dx in np.arange(-150, 151, 25):
                for dy in np.arange(-150, 151, 25):
                    t = t0 + np.array([dx, dy]); s_, ov = agreement(Q + t, PA, treeB, PB)
                    if s_ > bb[0]:
                        bb = (s_, th, t, ov)
        best[flip] = bb
    fl = max(best, key=lambda k: best[k][0]); s_, th, t, ov = best[fl]
    return dict(score=s_, flip=fl, theta=th, t=t, ca=ca, overlap=ov, score_other_flip=best[not fl][0])


PAIRED = {}
for f in sorted(glob.glob(os.path.join(P["PROC"], "sma", "*_paired.h5ad"))):
    M = ad.read_h5ad(f, backed="r"); PAIRED[M.uns["section"]] = (f, M.uns["visium_stem"], M.uns["mouse"], M.uns.get("brain_area", ""), float(M.uns.get("reg_iou", 0)))
if len(PAIRED) < 2:
    raise SystemExit("fewer than 2 registered sections in 02_processed/sma (run M13-01 / M13-02 first)")
VIS = {}; rows = []
for si, (fi, stem_i, mouse_i, area_i, iou_i) in PAIRED.items():
    for sj, (fj, stem_j, mouse_j, area_j, iou_j) in PAIRED.items():
        if si == sj or mouse_i != mouse_j or area_i != area_j or iou_i < 0.75 or iou_j < 0.75:
            continue
        t0 = time.time()
        for st in (stem_i, stem_j):
            if st not in VIS:
                VIS[st] = read_vis(st)
        A, B = VIS[stem_i], VIS[stem_j]; PA, PB = shared_pcs(A, B); R = register_vv(A, B, PA, PB)
        M = ad.read_h5ad(fi); q = np.column_stack([M.obs.x_vis_um.values, M.obs.y_vis_um.values])
        q2 = rot(q - R["ca"], R["flip"], R["theta"]) + R["t"]
        M.obs["x_vis_um"], M.obs["y_vis_um"] = q2[:, 0], q2[:, 1]
        cap = stem_j.split("_")[-1]; name = f"{si}x{cap}"
        M.uns.update(dict(section=name, msi_section=si, visium_stem=stem_j, visium_section_same=stem_i, pair_type="cross",
                          vv_score=R["score"], vv_score_other_flip=R["score_other_flip"], vv_flip=bool(R["flip"]), vv_theta_deg=float(np.rad2deg(R["theta"]) % 360),
                          vv_overlap=R["overlap"]))
        M.write_h5ad(os.path.join(OUT, f"{name}_paired.h5ad"))
        # reference for same-modality agreement: PC correlation of i -> j and the margin between the two flips
        rows.append(dict(pair=name, msi_section=si, visium_same=stem_i, visium_cross=stem_j, mouse=mouse_i, matrix=M.uns["matrix"],
                         vv_pc_corr=R["score"], vv_pc_corr_other_flip=R["score_other_flip"], flip=R["flip"], rotation_deg=np.rad2deg(R["theta"]) % 360,
                         overlap_frac=R["overlap"], seconds=time.time() - t0))
        C.log(f"[{si} MALDI -> {stem_j} Visium] Visium-Visium PC correlation {R['score']:.3f} (other flip {R['score_other_flip']:.3f}), flip {R['flip']}, "
              f"rotation {np.rad2deg(R['theta']) % 360:.1f} deg, overlap {R['overlap']:.0%}")
        fig, ax = plt.subplots(figsize=(4, 4))
        Qa = rot(A["xy"] - R["ca"], R["flip"], R["theta"]) + R["t"]
        ax.scatter(B["xy"][:, 0], B["xy"][:, 1], c=PB[:, 0], s=2, cmap="Greys", label=f"{stem_j} (PC1)")
        ax.scatter(Qa[:, 0], Qa[:, 1], c=PA[:, 0], s=1, cmap="viridis", alpha=.5, label=f"{stem_i} → {stem_j}")
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend(fontsize=5, markerscale=3); ax.set_title(f"{name}  r(PC) {R['score']:.2f}", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m15_vv_{name}.png"), dpi=150); plt.close(fig)
if not rows:
    raise SystemExit("no sections for cross pairs (needed: >= 2 sections of the same mouse and brain area with registration IoU >= 0.75)")
T = pd.DataFrame(rows); T.to_csv(os.path.join(TAB, "m15_cross_pairs.tsv"), sep="\t", index=False)
C.log("cross pairs:\n" + T.round(3).to_string(index=False) + f"\ndone ({time.time() - t00:.0f} s) -> {OUT}")
