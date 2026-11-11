#!/usr/bin/env python3

import argparse, os, sys, time
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.ndimage import map_coordinates, binary_dilation, gaussian_filter
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--res", type=float, default=50.0)
ap.add_argument("--cp-um", type=float, default=1000.0)
ap.add_argument("--max-disp-um", type=float, default=400.0)
ap.add_argument("--lam", type=float, default=0.02, help="weight of the smoothness penalty")
ap.add_argument("--maxiter", type=int, default=60)
ap.add_argument("--sim-rms-um", type=float, default=150.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); P = C.paths(a.root)
TAB = os.path.join(os.path.dirname(P["TAB"]), "Step2"); FIG = os.path.join(os.path.dirname(P["FIG"]), "Step2")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7})
TRAIN_BANDS = [(100, 300), (200, 600)]
EVAL_BANDS = [(50, 150), (100, 300), (200, 600), (400, 1200)]

pix = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"))
cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"))
ann = pd.read_csv(os.path.join(P["TAB"], "sbiad1426_peak_annotation.tsv"), sep="\t"); ann.index = pix.var_names
X = pix.X.tocsr(); tic = np.asarray(X.sum(1)).ravel()
tis = (~pix.obs["empty"].values.astype(bool)) & (tic > 0)
Xn = sp.diags(np.where(tis, np.median(tic[tis]) / np.maximum(tic, 1e-9), 0)) @ X
px, py = pix.obs.x_um.values, pix.obs.y_um.values
cx, cy = cel.obsm["spatial"][:, 0], cel.obsm["spatial"][:, 1]
x0, y0 = min(px[tis].min(), cx.min()), min(py[tis].min(), cy.min())
b = a.res
ixp, iyp = C.grid_bin(px[tis], py[tis], b, x0, y0); ixc, iyc = C.grid_bin(cx, cy, b, x0, y0)
nx = max(ixp.max(), ixc.max()) + 1; ny = max(iyp.max(), iyc.max()) + 1
lin = iyp * nx + ixp
Bm = sp.csr_matrix((np.ones(len(lin)), (lin, np.arange(len(lin)))), shape=(nx * ny, len(lin)))
cnt = np.asarray(Bm.sum(1)).ravel()
MET = (np.asarray((Bm @ Xn[tis]).todense()) / np.maximum(cnt, 1)[:, None]).reshape(ny, nx, -1)
cnt = cnt.reshape(ny, nx)
ncell = np.bincount(iyc * nx + ixc, minlength=nx * ny).reshape(ny, nx).astype(float)
mask = (cnt >= (b / 10) ** 2 / 3) & binary_dilation(ncell > 0, iterations=1)


def frac(lin_name):
    s = cel.obs["lineage"].values == lin_name
    return np.where(mask, np.bincount(iyc[s] * nx + ixc[s], minlength=nx * ny).reshape(ny, nx) / np.maximum(ncell, 1), np.nan)


def met_img(name):
    js = np.where(((ann.best_name == name) & ann.tier.isin(["A", "B"])).values)[0]
    return np.where(mask, np.log1p(MET[:, :, js]).mean(2), np.nan) if len(js) else None


REF = {"neuron": frac("Neuron"), "oligo": frac("Oligodendrocyte"), "malignant": frac("Malignant")}
PAIRS = {"NAA~neuron": ("N-acetylaspartate", "neuron"), "creatine~neuron": ("creatine", "neuron"),
         "GPC~oligo": ("glycerophosphocholine", "oligo"), "PCho~malignant": ("phosphocholine", "malignant")}
IMG = {k: (met_img(m), REF[r]) for k, (m, r) in PAIRS.items() if met_img(m) is not None}
TRAIN = "NAA~neuron"
TEST = [k for k in IMG if k != TRAIN]


def adduct_img(name, adduct):
    js = np.where(((ann.best_name == name) & (ann.best_adduct == adduct) & ann.tier.isin(["A", "B"])).values)[0]
    return np.where(mask, np.log1p(MET[:, :, js]).mean(2), np.nan) if len(js) else None


CTRL = {}
for nm, short in (("phosphocholine", "PCho"), ("glycerophosphocholine", "GPC"), ("creatine", "Cr")):
    m1, m2 = adduct_img(nm, "[M+H]+"), adduct_img(nm, "[M+K]+")
    if m1 is not None and m2 is not None:
        CTRL[f"{short} [M+H]+~[M+K]+"] = (m1, m2)
CTRL_TRAIN = next(iter(CTRL), None)
yy, xx = np.mgrid[0:ny, 0:nx]
cvb = int(round(1000 / b))
fold = ((xx // cvb) + (yy // cvb)) % 2
edge = np.zeros_like(mask)
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        edge |= np.roll(np.roll(fold, dy, 0), dx, 1) != fold
F0 = mask & (fold == 0); F1 = mask & (fold == 1) & ~edge
C.log(f"grid {ny}x{nx} ({b:.0f} µm); tissue bins {mask.sum()}; fold0 {F0.sum()} / fold1 (without border) {F1.sum()}; training pair {TRAIN}; test pairs {TEST}")

# ----------------------------------------------------------------------------- displacement field
gyc = int(np.ceil(ny * b / a.cp_um)) + 1; gxc = int(np.ceil(nx * b / a.cp_um)) + 1
cpy = np.linspace(0, gyc - 1, ny); cpx = np.linspace(0, gxc - 1, nx)
CY, CX = np.meshgrid(cpy, cpx, indexing="ij")


def field(theta):
    d = theta.reshape(2, gyc, gxc)
    dy = map_coordinates(d[0], [CY, CX], order=3, mode="nearest"); dx = map_coordinates(d[1], [CY, CX], order=3, mode="nearest")
    return dy, dx


def warp(img, dy, dx):
    m_ = np.isfinite(img).astype(float); v = np.where(np.isfinite(img), img, 0.0)
    wv = map_coordinates(v, [yy + dy, xx + dx], order=1, cval=0.0); wm = map_coordinates(m_, [yy + dy, xx + dx], order=1, cval=0.0)
    return np.where((wm > 0.5) & mask, wv / np.maximum(wm, 1e-6), np.nan)


def band_r(A_, B_, region, lo, hi):
    Ab, Bb = C.band(A_, lo, hi, b), C.band(B_, lo, hi, b)
    return C.rr(np.where(region, Ab, np.nan), np.where(region, Bb, np.nan))


def evaluate(imgs, dy, dx, region, tag, train=TRAIN):
    rows = []
    for k, (Mi, Ri) in imgs.items():
        W = warp(Mi, dy, dx)
        for lo, hi in EVAL_BANDS:
            rows.append(dict(condition=tag, pair=k, role="train" if k == train else "test", band=f"{lo}-{hi}", band_lo=lo, r=band_r(W, Ri, region, lo, hi)))
    return rows


def fit(imgs, region, train=TRAIN):
    """During optimization the images are band-passed first and then deformed (approximately commutative for small smooth deformations),
    so each objective evaluation needs one interpolation and one correlation; evaluation uses deformation followed by band-pass."""
    Mi, Ri = imgs[train]
    lim = a.max_disp_um / b
    pre = []
    for lo, hi in TRAIN_BANDS:
        Mb = C.band(Mi, lo, hi, b); Rb = np.where(region, C.band(Ri, lo, hi, b), np.nan)
        pre.append((np.where(np.isfinite(Mb), Mb, 0.0), np.isfinite(Mb).astype(float), Rb))
    okR = [np.isfinite(Rb) for _, _, Rb in pre]

    def obj(theta):
        dy, dx = field(theta)
        coords = [yy + dy, xx + dx]
        rs = []
        for (Mv, Mm, Rb), ok in zip(pre, okR):
            wv = map_coordinates(Mv, coords, order=1, cval=0.0); wm = map_coordinates(Mm, coords, order=1, cval=0.0)
            sel = ok & (wm > 0.5)
            if sel.sum() > 30:
                u = wv[sel] / wm[sel]; v = Rb[sel]
                rs.append(np.corrcoef(u, v)[0, 1])
        r = np.mean(rs) if rs else 0.0
        d = theta.reshape(2, gyc, gxc)
        smooth = sum(np.sum(np.diff(d[c], axis=ax) ** 2) for c in (0, 1) for ax in (0, 1)) / d.size
        return -r + a.lam * smooth
    t0 = time.time()
    res = minimize(obj, np.zeros(2 * gyc * gxc), method="L-BFGS-B", bounds=[(-lim, lim)] * (2 * gyc * gxc),
                   options=dict(maxiter=a.maxiter, eps=0.5))
    C.log(f"    optimization: {res.nit} iterations, {time.time() - t0:.0f} s, objective {obj(np.zeros_like(res.x)):.3f} -> {res.fun:.3f}")
    return res.x


rows = []
z = np.zeros(2 * gyc * gxc); dy0, dx0 = field(z)
C.log("== real data")
rows += evaluate(IMG, dy0, dx0, F1, "original")
th = fit(IMG, F0); dyR, dxR = field(th)
rows += evaluate(IMG, dyR, dxR, F1, "refined")
disp = np.hypot(dyR, dxR)[mask] * b
C.log(f"    refined displacement: median {np.median(disp):.0f} µm, 95th percentile {np.percentile(disp, 95):.0f} µm")

C.log(f"== positive control (MALDI adduct pairs, smooth deformation of RMS {a.sim_rms_um:.0f} µm): training {CTRL_TRAIN}, test {[k for k in CTRL if k != CTRL_TRAIN]}")
rng = np.random.default_rng(a.seed)
sim = gaussian_filter(rng.normal(size=(2, gyc, gxc)), 0.8)
sim *= (a.sim_rms_um / b) / (np.sqrt(np.mean(sim ** 2)) + 1e-9)
dyS, dxS = field(sim.ravel())
rows_c = evaluate(CTRL, dy0, dx0, F1, "ctrl_original", CTRL_TRAIN)
CTRL_S = {k: (warp(Mi, dyS, dxS), Ri) for k, (Mi, Ri) in CTRL.items()}
rows_c += evaluate(CTRL_S, dy0, dx0, F1, "ctrl_simulated_misreg", CTRL_TRAIN)
thS = fit(CTRL_S, F0, CTRL_TRAIN); dyS2, dxS2 = field(thS)
rows_c += evaluate(CTRL_S, dyS2, dxS2, F1, "ctrl_simulated_refined", CTRL_TRAIN)
rows += rows_c
R = pd.DataFrame(rows); R.to_csv(os.path.join(TAB, "sbiad_refinement_test.tsv"), sep="\t", index=False)

piv = R[R.role == "test"].groupby(["condition", "band_lo"]).r.mean().unstack("condition")
fine = [50, 100]                                                  # the 50-150 and 100-300 µm bands
d_real = float((piv.loc[fine, "refined"] - piv.loc[fine, "original"]).mean())
lost = float((piv.loc[fine, "ctrl_original"] - piv.loc[fine, "ctrl_simulated_misreg"]).mean())
rec = float((piv.loc[fine, "ctrl_simulated_refined"] - piv.loc[fine, "ctrl_simulated_misreg"]).mean())
recovery = rec / lost if lost > 0.05 else np.nan
C.log(f"    real cross-modal test pairs: fine-scale delta r = {d_real:+.3f}; positive control: loss from deformation {lost:.3f}, recovered {rec:.3f} ({recovery if np.isfinite(recovery) else float('nan'):.0%})")
if d_real >= 0.10:
    verdict = "A: registration-limited; refinement improves fine-scale agreement on independent test pairs; M3 uses the refined coordinates"
elif d_real < 0.05 and np.isfinite(recovery) and recovery >= 0.5:
    verdict = (f"B: limit of adjacent sections; the method recovers {recovery:.0%} of the loss caused by simulated misregistration, but on real data the fine-scale delta r is only {d_real:+.3f}; "
               "fine-scale information is not lost by registration but differs between adjacent sections -> the unit of analysis is a metabolic neighbourhood of >= 200 µm")
else:
    verdict = f"undetermined: real delta r {d_real:+.3f}, simulated recovery {recovery if np.isfinite(recovery) else float('nan'):.0%}; check --lam / --cp-um"
out = dict(verdict=verdict, delta_r_test_100_300=d_real, simulated_loss=lost, simulated_recovered=rec, recovery_fraction=recovery,
           refined_displacement_um=dict(median=float(np.median(disp)), p95=float(np.percentile(disp, 95))),
           params=dict(cp_um=a.cp_um, lam=a.lam, max_disp_um=a.max_disp_um, sim_rms_um=a.sim_rms_um))
C.save_json(out, os.path.join(TAB, "sbiad_refinement_test.json"))
np.savez_compressed(os.path.join(P["PROC"], "sbiad1426_refined_displacement.npz"), dy=dyR * b, dx=dxR * b, x0=x0, y0=y0, res=b,
                    use_refined=bool(d_real >= 0.10))
C.log(f"decision: {verdict}")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for cond, st, col in (("original", "o-", "#c0392b"), ("refined", "s--", "#c0392b"),
                      ("ctrl_original", "o-", "#2c7fb8"), ("ctrl_simulated_misreg", "v:", "#2c7fb8"), ("ctrl_simulated_refined", "^-.", "#2c7fb8")):
    d = R[(R.condition == cond) & (R.role == "test")].groupby("band_lo").r.mean()
    axs[0].plot(range(len(d)), d.values, st, color=col, label=cond.replace("ctrl_", "MALDI adduct ctrl: "))
axs[0].set_xticks(range(len(EVAL_BANDS))); axs[0].set_xticklabels([f"{l_}-{h_}" for l_, h_ in EVAL_BANDS], rotation=30)
axs[0].set_ylabel("mean r (test pairs, held-out fold)"); axs[0].legend(fontsize=5); axs[0].set_title("red: real cross-modal; blue: positive control")
im = axs[1].imshow(np.where(mask, np.hypot(dyR, dxR) * b, np.nan), cmap="magma"); fig.colorbar(im, ax=axs[1], label="µm")
axs[1].set_title("learned displacement (real data)"); axs[1].axis("off")
im = axs[2].imshow(np.where(mask, np.hypot(dyS, dxS) * b, np.nan), cmap="magma"); fig.colorbar(im, ax=axs[2], label="µm")
axs[2].set_title("simulated displacement (positive control)"); axs[2].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "sbiad_refinement_test.png"), dpi=150); plt.close(fig)
