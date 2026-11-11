#!/usr/bin/env python3

import argparse, glob, os, sys
import numpy as np, pandas as pd, anndata as ad
from scipy.ndimage import binary_closing, binary_fill_holes, label
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--expect", type=int, default=6)
ap.add_argument("--close-iter", type=int, default=3)
ap.add_argument("--min-frac", type=float, default=0.02)
a = ap.parse_args(); P = C.paths(a.root)
TAB = os.path.join(os.path.dirname(P["TAB"]), "Step2"); FIG = os.path.join(os.path.dirname(P["FIG"]), "Step2")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(P["PROC"], "ravi_maldi_sections"); os.makedirs(OUT, exist_ok=True)

fs = sorted(glob.glob(os.path.join(P["PROC"], "ravi_maldi", "*.h5ad")))
if not fs:
    raise SystemExit("no 02_processed/ravi_maldi/*.h5ad; run run_step1.sh 3-4 first")
A = ad.read_h5ad(fs[0])
src = A.uns.get("source", {})
if "pre-cropped to tissue" not in str(src.get("tissue_mask", "")):
    C.log("[warning] the h5ad was not built with the tissue-preserving mask of M1-03; rebuild it with run_step1.sh 3-4")
px_um = float(A.uns.get("pixel_size_um", 50) or 50)
x = A.obs.x_px.values.astype(int); y = A.obs.y_px.values.astype(int)
x0, y0 = x.min(), y.min(); gx, gy = x - x0, y - y0
G = np.zeros((gy.max() + 1, gx.max() + 1), bool)
tis = A.obs["tissue"].values.astype(bool) & (A.obs["tic"].values > 0)
G[gy[tis], gx[tis]] = True
Gc = binary_fill_holes(binary_closing(G, iterations=a.close_iter))
lab, n = label(Gc)
sizes = np.bincount(lab.ravel()); sizes[0] = 0
keep = np.where(sizes >= a.min_frac * sizes.sum())[0]
C.log(f"{n} connected components, {len(keep)} kept (>= {a.min_frac:.0%} of the area)")
if len(keep) != a.expect:
    C.log(f"[warning] {a.expect} sections expected, {len(keep)} found; check ravi_maldi_sections.png and adjust --close-iter if needed")
cent = {k: (np.mean(np.where(lab == k)[1]), np.mean(np.where(lab == k)[0])) for k in keep}
# order by row, then column: rows clustered by y (gap > half the mean section height), x within rows
ys = sorted(keep, key=lambda k: cent[k][1])
rows, cur = [], [ys[0]]
hmean = np.mean([np.ptp(np.where(lab == k)[0]) for k in keep])
for k in ys[1:]:
    if cent[k][1] - np.mean([cent[j][1] for j in cur]) > hmean / 2:
        rows.append(cur); cur = [k]
    else:
        cur.append(k)
rows.append(cur)
order = [k for r in rows for k in sorted(r, key=lambda j: cent[j][0])]
sec_of_px = lab[gy, gx]
recs = []
for i, k in enumerate(order, 1):
    sel = (sec_of_px == k) & tis
    S = A[sel].copy()
    S.obs["section"] = f"S{i}"
    S.obs["x_local_um"] = (S.obs.x_px.values - S.obs.x_px.min()) * px_um
    S.obs["y_local_um"] = (S.obs.y_px.values - S.obs.y_px.min()) * px_um
    S.obsm["spatial"] = S.obs[["x_local_um", "y_local_um"]].values.astype(float)
    S.uns["section"] = dict(id=f"S{i}", component=int(k), centroid_px=[float(cent[k][0] + x0), float(cent[k][1] + y0)],
                            parent=os.path.basename(fs[0]), row=int(next(r for r, rr_ in enumerate(rows) if k in rr_)) + 1)
    S.write_h5ad(os.path.join(OUT, f"S{i}.h5ad"), compression="gzip")
    w_um = np.ptp(S.obs.x_px) * px_um; h_um = np.ptp(S.obs.y_px) * px_um
    recs.append(dict(section=f"S{i}", n_pixels=int(sel.sum()), area_mm2=sel.sum() * (px_um / 1000) ** 2, width_mm=w_um / 1000, height_mm=h_um / 1000,
                     grid_row=S.uns["section"]["row"], centroid_x_px=cent[k][0] + x0, centroid_y_px=cent[k][1] + y0,
                     median_tic=float(np.median(S.obs.tic))))
    C.log(f"  S{i}: {sel.sum():,} pixels, {recs[-1]['area_mm2']:.1f} mm2, {w_um / 1000:.1f} x {h_um / 1000:.1f} mm")
T = pd.DataFrame(recs); T.to_csv(os.path.join(TAB, "ravi_maldi_sections.tsv"), sep="\t", index=False)

fig, axx = plt.subplots(figsize=(10, 10 * G.shape[0] / G.shape[1] + 0.5))
im = np.full(G.shape, np.nan)
for i, k in enumerate(order, 1):
    im[(lab == k) & G] = i
axx.imshow(im, cmap="tab10", vmin=0.5, vmax=10.5)
for i, k in enumerate(order, 1):
    axx.text(cent[k][0], cent[k][1], f"S{i}", ha="center", va="center", fontsize=14, weight="bold", color="k")
axx.set_title(f"Ravi 2022 combined MALDI → {len(order)} sections (README patients: UKF275/262/260/259/248/296_T; identity resolved in M2-03)")
axx.axis("off"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "ravi_maldi_sections.png"), dpi=130); plt.close(fig)
C.log(f"-> {OUT}/S*.h5ad; {TAB}/ravi_maldi_sections.tsv")
