#!/usr/bin/env python3

import argparse, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from scipy.stats import spearmanr
from scipy.ndimage import binary_dilation, gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--bins", default="50,100,200")
ap.add_argument("--block-um", type=float, default=500.0, help="block size (µm) of the block permutation; larger than the autocorrelation length")
ap.add_argument("--n-perm", type=int, default=999)
ap.add_argument("--max-shift-um", type=float, default=600.0)
ap.add_argument("--cv-block-um", type=float, default=1000.0, help="checkerboard tile size (µm) of the spatial cross-validation")
ap.add_argument("--top-k", type=int, default=5, help="metabolites selected per direction in the cross-validation")
a = ap.parse_args(); P = C.paths(a.root)
plt.rcParams.update({"font.size": 7})
BANDS = [(25, 100), (50, 150), (100, 300), (200, 600), (400, 1200), (800, None)]

pix = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"))
cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"))
ann = pd.read_csv(os.path.join(P["TAB"], "sbiad1426_peak_annotation.tsv"), sep="\t")
tg = pd.read_csv(os.path.join(P["TAB"], "sbiad1426_targets.tsv"), sep="\t")
ann.index = pix.var_names
use = ann["use_downstream"].values.astype(bool)
bn = ann.best_name.fillna("").astype(str); ba = ann.best_adduct.fillna("").astype(str)
ann["label"] = np.where(bn != "", bn + " " + ba, "m/z " + ann.mz_cal.round(4).astype(str))
C.log(f"{pix.n_obs:,} pixels; {cel.n_obs:,} cells; peaks for downstream use {use.sum()}/{len(use)}")

# ----------------------------------------------------------------------------- TIC normalization of pixels
X = pix.X.tocsr()
tic = np.asarray(X.sum(1)).ravel()
tissue = (~pix.obs["empty"].values.astype(bool)) & (tic > 0)
Xn = sp.diags(np.where(tissue, np.median(tic[tissue]) / np.maximum(tic, 1e-9), 0.0)) @ X
px, py = pix.obs["x_um"].values, pix.obs["y_um"].values
cx, cy = cel.obsm["spatial"][:, 0], cel.obsm["spatial"][:, 1]
x0 = min(px[tissue].min(), cx.min()); y0 = min(py[tissue].min(), cy.min())
lin = cel.obs["lineage"].values
anno = cel.obs["Anno"].astype(str).values


def peak_idx(name, adducts=None, tiers=("A", "B")):
    m = (ann.best_name == name) & ann.tier.isin(tiers)
    if adducts:
        m &= ann.best_adduct.isin(adducts)
    return np.where(m.values)[0]


def build_grid(b):
    ixp, iyp = C.grid_bin(px[tissue], py[tissue], b, x0, y0)
    ixc, iyc = C.grid_bin(cx, cy, b, x0, y0)
    nx = max(ixp.max(), ixc.max()) + 1; ny = max(iyp.max(), iyc.max()) + 1
    linp = iyp * nx + ixp
    Bm = sp.csr_matrix((np.ones(len(linp)), (linp, np.arange(len(linp)))), shape=(nx * ny, len(linp)))
    cnt = np.asarray(Bm.sum(1)).ravel()
    met = (np.asarray((Bm @ Xn[tissue]).todense()) / np.maximum(cnt, 1)[:, None]).reshape(ny, nx, -1)
    cnt = cnt.reshape(ny, nx)
    types = {}
    for nm, lab in (("lineage", lin), ("anno", anno)):
        for t in np.unique(lab):
            s = lab == t
            types[f"{nm}:{t}"] = np.bincount(iyc[s] * nx + ixc[s], minlength=nx * ny).reshape(ny, nx).astype(float)
    ncell = np.bincount(iyc * nx + ixc, minlength=nx * ny).reshape(ny, nx).astype(float)
    mask = (cnt >= max(1, (b / 10) ** 2 / 3)) & binary_dilation(ncell > 0, iterations=1)   # MALDI pixels are 10 µm
    frac = lambda key: np.where(mask, types.get(key, np.zeros_like(ncell)) / np.maximum(ncell, 1), np.nan)
    return dict(met=met, cnt=cnt, types=types, ncell=ncell, mask=mask, frac=frac)


def zmap(img):
    return (img - np.nanmean(img)) / (np.nanstd(img) + 1e-12)


def nconv(img, sig):
    m_ = np.isfinite(img); v = np.where(m_, img, 0.0)
    return gaussian_filter(v, sig) / np.maximum(gaussian_filter(m_.astype(float), sig), 1e-6)


def band(img, lo, hi, b):
    out = nconv(img, lo / b) - (nconv(img, hi / b) if hi else 0.0)
    return np.where(np.isfinite(img), out, np.nan)


def rr(A_, B_):
    ok = np.isfinite(A_) & np.isfinite(B_)
    return float(np.corrcoef(A_[ok], B_[ok])[0, 1]) if ok.sum() > 30 else np.nan


def shift_corr(A_, B_, sx, sy):
    ny, nx = A_.shape
    a_ = A_[max(0, sy):ny + min(0, sy), max(0, sx):nx + min(0, sx)]
    b_ = B_[max(0, -sy):ny + min(0, -sy), max(0, -sx):nx + min(0, -sx)]
    return rr(a_, b_)


bins = [float(b) for b in a.bins.split(",")]
res = dict(bins_um=bins, scales={})
maps50 = None
for b in bins:
    C.log(f"== bins of {b:.0f} µm")
    g = build_grid(b); M = g["mask"]
    oligo, neuro = g["frac"]("lineage:Oligodendrocyte"), g["frac"]("lineage:Neuron")
    axis = zmap(oligo) - zmap(neuro)                           # white matter (+) vs grey matter (-)
    yy, xx = np.where(M)
    L = np.log1p(g["met"][yy, xx])
    img_of = lambda v: (lambda im: (im.__setitem__((yy, xx), v), im)[1])(np.full(M.shape, np.nan))
    block = max(2, int(round(a.block_um / b)))
    sc = dict(n_bins=int(M.sum()), block_bins=block)

    # ---- (2) pre-specified positive controls
    pre = []
    for nm, ref, refname in (("N-acetylaspartate", neuro, "neuron fraction"), ("creatine", neuro, "neuron fraction"),
                             ("glycerophosphocholine", oligo, "oligodendrocyte fraction"), ("cholesterol", oligo, "oligodendrocyte fraction")):
        js = peak_idx(nm)
        if len(js) == 0:
            continue
        s = L[:, js].mean(1)
        r, p, _ = C.block_perm_test(s, ref[yy, xx], xx, yy, block=block, n_perm=a.n_perm)
        pre.append(dict(metabolite=nm, peaks=";".join(ann.label.values[js]), reference=refname, rho=float(r[0]),
                        p_block_perm=float(p[0]), p_naive=float(spearmanr(s, ref[yy, xx]).pvalue)))
        C.log(f"   control {nm:<22} vs {refname:<26} rho = {r[0]:+.3f}  p_block = {p[0]:.3g}  (parametric p = {pre[-1]['p_naive']:.1e})")
    sc["prespecified_controls"] = pre

    # ---- (3) spatial checkerboard cross-validation
    cvb = max(2, int(round(a.cv_block_um / b)))
    fold = ((xx // cvb) + (yy // cvb)) % 2
    F = np.full(M.shape, -1); F[yy, xx] = fold
    edge = np.zeros(len(xx), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nb = np.full(M.shape, -1)
            nb[max(0, dy):M.shape[0] + min(0, dy), max(0, dx):M.shape[1] + min(0, dx)] = \
                F[max(0, -dy):M.shape[0] + min(0, -dy), max(0, -dx):M.shape[1] + min(0, -dx)]
            edge |= (nb[yy, xx] >= 0) & (nb[yy, xx] != fold)
    cand = np.where(use)[0]
    ho = []
    for k in (0, 1):
        train, test = fold == k, (fold != k) & ~edge
        rho = np.nan_to_num(np.array([spearmanr(L[train, j], axis[yy[train], xx[train]])[0] for j in cand]))
        wm = cand[np.argsort(-rho)[:a.top_k]]; gm = cand[np.argsort(rho)[:a.top_k]]
        Z = (L - L[train].mean(0)) / (L[train].std(0) + 1e-9)
        score = Z[:, wm].mean(1) - Z[:, gm].mean(1)
        r, p, _ = C.block_perm_test(score[test], axis[yy[test], xx[test]], xx[test], yy[test], block=block, n_perm=a.n_perm)
        ho.append(dict(direction=f"fold{k}→fold{1 - k}", rho_test=float(r[0]), p_block_perm=float(p[0]), n_test=int(test.sum()),
                       wm_metabolites=";".join(ann.label.values[wm]), gm_metabolites=";".join(ann.label.values[gm])))
        C.log(f"   cross-validation fold{k}->fold{1 - k}: {test.sum()} test bins, rho = {r[0]:+.3f}, p_block = {p[0]:.3g}"
              f" | white-matter side {', '.join(ann.label.values[wm][:3])} ... | grey-matter side {', '.join(ann.label.values[gm][:3])} ...")
    sc["held_out"] = ho

    # anatomical score of all metabolites (maps / orientation check)
    rho_all = np.nan_to_num(np.array([spearmanr(L[:, j], axis[yy, xx])[0] for j in cand]))
    wm = cand[np.argsort(-rho_all)[:a.top_k]]; gm = cand[np.argsort(rho_all)[:a.top_k]]
    Z = (L - L.mean(0)) / (L.std(0) + 1e-9)
    mscore = img_of(Z[:, wm].mean(1) - Z[:, gm].mean(1))
    pd.DataFrame(dict(peak=ann.index[cand], label=ann.label.values[cand], tier=ann.tier.values[cand], rho_with_WM_GM_axis=rho_all)) \
        .sort_values("rho_with_WM_GM_axis").to_csv(os.path.join(P["TAB"], f"sbiad1426_metabolite_vs_anatomy_{int(b)}um.tsv"), sep="\t", index=False)

    # ---- (4) orientation check
    orient = {nm_: rr(f(mscore), axis) for nm_, f in (("identity", lambda z: z), ("flip_x", lambda z: z[:, ::-1]),
                                                       ("flip_y", lambda z: z[::-1, :]), ("rot180", lambda z: z[::-1, ::-1]))}
    sc["orientation_r"] = orient
    C.log("   orientation: " + ", ".join(f"{k} {v:+.2f}" for k, v in orient.items()))

    # ---- (8) WEHI positive control
    js = peak_idx("glycerophosphocholine")
    if len(js):
        opc_like = g["frac"]("anno:OPC-like")
        s = L[:, js].mean(1)
        r, p, _ = C.block_perm_test(s, opc_like[yy, xx], xx, yy, block=block, n_perm=a.n_perm)
        resid = s - np.polyval(np.polyfit(axis[yy, xx], s, 1), axis[yy, xx])     # adjusted for white / grey matter
        r2, p2, _ = C.block_perm_test(resid, opc_like[yy, xx], xx, yy, block=block, n_perm=a.n_perm)
        sc["WEHI_control_GPC_vs_OPC_like"] = dict(rho=float(r[0]), p_block=float(p[0]), rho_adj_anatomy=float(r2[0]), p_block_adj=float(p2[0]))
        C.log(f"   WEHI control GPC vs OPC-like fraction: rho = {r[0]:+.3f} (p_block {p[0]:.3g}); adjusted for white matter rho = {r2[0]:+.3f} (p_block {p2[0]:.3g})")
    res["scales"][f"{int(b)}um"] = sc
    if maps50 is None:
        maps50 = dict(b=b, g=g, L=L, yy=yy, xx=xx, axis=axis, mscore=mscore, oligo=oligo, neuro=neuro)

    # ---- maps
    show = [("oligodendrocyte fraction", oligo), ("neuron fraction", neuro), ("WM−GM axis (Xenium)", axis), ("metabolite WM−GM score", mscore)]
    for nm in ("N-acetylaspartate", "creatine", "glycerophosphocholine", "phosphocholine"):
        js = peak_idx(nm)
        if len(js):
            show.append((nm + " (MALDI)", img_of(L[:, js].mean(1))))
    show += [("OPC-like fraction", g["frac"]("anno:OPC-like")), ("malignant fraction", g["frac"]("lineage:Malignant"))]
    nc = 5; nr = int(np.ceil(len(show) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.8 * nr))
    for axx, (t, im) in zip(axs.ravel(), show):
        lo_, hi_ = np.nanpercentile(im, [2, 98]) if np.isfinite(im).any() else (0, 1)
        axx.imshow(im, cmap="viridis", vmin=lo_, vmax=hi_); axx.set_title(t); axx.axis("off")
    for axx in axs.ravel()[len(show):]:
        axx.axis("off")
    fig.suptitle(f"S-BIAD1426 · {b:.0f} µm bins · Xenium cell-type maps vs MALDI (adjacent section)")
    fig.tight_layout(); fig.savefig(os.path.join(P["FIG"], f"sbiad1426_registration_maps_{int(b)}um.png"), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------- (6) scale-resolved concordance + (7) shift scan
C.log(f"== scale-resolved concordance (band-pass on the {maps50['b']:.0f}-µm grid)")
b = maps50["b"]; g = maps50["g"]; L = maps50["L"]; yy, xx = maps50["yy"], maps50["xx"]
img_of = lambda v: (lambda im: (im.__setitem__((yy, xx), v), im)[1])(np.full(g["mask"].shape, np.nan))
cross = {}
for nm, ref in (("N-acetylaspartate", "lineage:Neuron"), ("creatine", "lineage:Neuron"),
                ("glycerophosphocholine", "lineage:Oligodendrocyte"), ("phosphocholine", "lineage:Malignant")):
    js = peak_idx(nm)
    if len(js):
        cross[f"{nm} ~ {ref.split(':')[1]}"] = (img_of(L[:, js].mean(1)), g["frac"](ref))
cross["metabolite score ~ WM−GM axis"] = (maps50["mscore"], maps50["axis"])
ceil = {}
for nm in ann.best_name[ann.tier == "A"].unique():
    js = peak_idx(nm, tiers=("A",))
    if len(js) >= 2:
        js = js[np.argsort(-ann.frac_pixels.values[js])][:2]
        ceil[f"{nm}: {ann.best_adduct.values[js[0]]} ~ {ann.best_adduct.values[js[1]]}"] = (img_of(L[:, js[0]]), img_of(L[:, js[1]]))
rows = []
for lo, hi in BANDS:
    tag = f"{lo}-{hi if hi else 'inf'}"
    for kind, dct in (("cross-modal", cross), ("MALDI ceiling", ceil)):
        for pn, (A_, B_) in dct.items():
            rows.append(dict(band_um=tag, band_lo=lo, kind=kind, pair=pn, r=rr(band(A_, lo, hi, b), band(B_, lo, hi, b))))
spec = pd.DataFrame(rows)
spec.to_csv(os.path.join(P["TAB"], "sbiad1426_scale_spectrum.tsv"), sep="\t", index=False)
summ = spec.groupby(["band_lo", "band_um", "kind"]).r.median().unstack("kind").reset_index().sort_values("band_lo")
for _, r_ in summ.iterrows():
    C.log(f"   band {r_['band_um']:>9} µm: median cross-modal r = {r_.get('cross-modal', np.nan):.2f}   "
          f"median MALDI ceiling (adduct pairs) r = {r_.get('MALDI ceiling', np.nan):.2f}")
cm = summ.set_index("band_lo")["cross-modal"]
eff = [lo for lo, v in cm.items() if v >= 0.3]
eff_band = summ.set_index("band_lo").loc[eff[0], "band_um"] if eff else None
C.log(f"   effective resolution: finest band with cross-modal r >= 0.3 = {eff_band} µm")

# shift scan on the effective band (or 200-600 µm)
lo, hi = next(((l_, h_) for l_, h_ in BANDS if f"{l_}-{h_ if h_ else 'inf'}" == eff_band), (200, 600))
hi = hi or 1200
ms = int(round(a.max_shift_um / b)); shifts = range(-ms, ms + 1)
dist = np.hypot(*np.meshgrid(np.arange(-ms, ms + 1), np.arange(-ms, ms + 1))) * b
scan = {}
for pn in ("N-acetylaspartate ~ Neuron", "metabolite score ~ WM−GM axis"):
    if pn not in cross:
        continue
    A_, B_ = (band(z, lo, hi, b) for z in cross[pn])
    S = np.array([[shift_corr(A_, B_, sx, sy) for sx in shifts] for sy in shifts])
    iy_, ix_ = np.unravel_index(np.nanargmax(S), S.shape)
    border = ix_ in (0, 2 * ms) or iy_ in (0, 2 * ms)
    off = None if border else (float((ix_ - ms) * b), float((iy_ - ms) * b))
    ring = pd.Series(S.ravel()).groupby(np.round(dist.ravel() / b) * b).mean()
    half = [float(d) for d, v in ring.items() if d > 0 and v < S[ms, ms] / 2]
    scan[pn] = dict(band_um=f"{lo}-{hi}", r_at_zero=float(S[ms, ms]), r_max=float(np.nanmax(S)), best_offset_um=off,
                    half_decay_distance_um=(half[0] if half else None), surface=S, ring=ring)
    C.log(f"   shift scan [{pn}, band {lo}-{hi} µm]: r(0) = {S[ms, ms]:.3f}, maximum {np.nanmax(S):.3f} at offset "
          f"{off if off is not None else 'scan border (undetermined)'} µm; correlation halved at about {half[0] if half else '>' + str(a.max_shift_um)} µm")
# a residual offset is accepted only if both pairs agree
offs = [v["best_offset_um"] for v in scan.values() if v["best_offset_um"] is not None and v["r_max"] - v["r_at_zero"] >= 0.05]
consistent = len(offs) == len(scan) and len(offs) >= 2 and np.hypot(offs[0][0] - offs[1][0], offs[0][1] - offs[1][1]) <= lo
res_offset = tuple(np.mean(offs, 0)) if consistent else None
C.log(f"   residual offset: {'consistent between pairs, about %s µm' % (tuple(round(v) for v in res_offset),) if consistent else 'pairs disagree or gain < 0.05: no reliable evidence of a residual offset'}")
res["scale_spectrum"] = dict(median_by_band=summ.to_dict(orient="records"), effective_band_um=eff_band)
res["shift_scan"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("surface", "ring")} for k, v in scan.items()}

fig, axs = plt.subplots(1, 2 + len(scan), figsize=(4.2 * (2 + len(scan)), 3.4))
pos = {l_: i for i, (l_, _) in enumerate(BANDS)}
for kind, col in (("cross-modal", "#c0392b"), ("MALDI ceiling", "#2c7fb8")):
    s_ = spec[spec.kind == kind]
    for pn, d in s_.groupby("pair"):
        d = d.sort_values("band_lo"); axs[0].plot([pos[v] for v in d.band_lo], d.r, "-", color=col, alpha=.35, lw=1)
    m_ = s_.groupby("band_lo").r.median(); axs[0].plot([pos[v] for v in m_.index], m_.values, "o-", color=col, lw=2.5, label=kind + " (median)")
axs[0].set_xticks(range(len(BANDS))); axs[0].set_xticklabels([f"{l_}-{h_ or '∞'}" for l_, h_ in BANDS], rotation=30)
axs[0].axhline(0.3, ls="--", c="grey"); axs[0].set_xlabel("spatial band (µm)"); axs[0].set_ylabel("Pearson r"); axs[0].legend()
axs[0].set_title("scale-resolved concordance")
for pn, v in scan.items():
    axs[1].plot(v["ring"].index, v["ring"].values, "o-", ms=3, label=pn)
axs[1].set_xlabel("shift distance (µm)"); axs[1].set_ylabel("mean r"); axs[1].legend(fontsize=6); axs[1].set_title(f"decay with misalignment (band {lo}-{hi} µm)")
for axx, (pn, v) in zip(axs[2:], scan.items()):
    im = axx.imshow(v["surface"], extent=[-ms * b - b / 2, ms * b + b / 2, ms * b + b / 2, -ms * b - b / 2], cmap="magma")
    if v["best_offset_um"] is not None:
        axx.plot(*v["best_offset_um"], "c+", ms=12)
    axx.set_xlabel("shift x (µm)"); axx.set_ylabel("shift y (µm)"); fig.colorbar(im, ax=axx); axx.set_title(pn, fontsize=7)
fig.suptitle("S-BIAD1426: effective resolution of MALDI↔Xenium coupling"); fig.tight_layout()
fig.savefig(os.path.join(P["FIG"], "sbiad1426_scale_spectrum_shift_scan.png"), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------- cell type x metabolite fingerprints (descriptive)
C.log("== cell-type metabolic fingerprints (descriptive; adjacent-section mapping represents the neighbourhood of a cell)")
Mm = cel.obsm["metab"]; mt = Mm.sum(1)
Mn = np.log1p(Mm / np.maximum(mt, 1)[:, None] * np.median(mt[mt > 0]))
if not np.allclose(np.asarray(cel.uns["metab_mz"], float), pix.var["mz"].values, atol=1e-3):
    C.log("  [warning] m/z order differs between cell- and pixel-level data; fingerprint plot skipped");
else:
    cols = np.where(ann.tier.isin(["A", "B"]).values & use)[0]
    types = pd.Series(anno).value_counts(); types = types[types >= 50].index
    prof = pd.DataFrame({t: Mn[anno == t][:, cols].mean(0) for t in types}, index=ann.label.values[cols]).T
    z = (prof - prof.mean()) / (prof.std() + 1e-9)
    z.to_csv(os.path.join(P["TAB"], "sbiad1426_celltype_metabolite_z.tsv"), sep="\t")
    fig, axx = plt.subplots(figsize=(0.28 * z.shape[1] + 3, 0.28 * z.shape[0] + 2))
    im = axx.imshow(z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    axx.set_xticks(range(z.shape[1])); axx.set_xticklabels(z.columns, rotation=90); axx.set_yticks(range(z.shape[0])); axx.set_yticklabels(z.index)
    fig.colorbar(im, ax=axx, shrink=.5, label="z across cell types")
    axx.set_title("S-BIAD1426 cell-type metabolite fingerprints (descriptive; neighbourhood-level)")
    fig.tight_layout(); fig.savefig(os.path.join(P["FIG"], "sbiad1426_celltype_metabolite_heatmap.png"), dpi=160); plt.close(fig)

# ----------------------------------------------------------------------------- decision
key = f"{int(bins[1] if len(bins) > 1 else bins[0])}um"
sk = res["scales"].get(key, {})
naa = [c for c in sk.get("prespecified_controls", []) if c["metabolite"] == "N-acetylaspartate"]
naa_ok = bool(naa) and naa[0]["rho"] >= 0.3 and naa[0]["p_block_perm"] < 0.05
cv_ok = bool(sk.get("held_out")) and all(h["rho_test"] >= 0.3 and h["p_block_perm"] < 0.05 for h in sk["held_out"])
orr = sk.get("orientation_r", {})
orient_ok = bool(orr) and orr["identity"] - max(v for k_, v in orr.items() if k_ != "identity") >= 0.2
off_d = float(np.hypot(*res_offset)) if res_offset is not None else float("nan")
res["residual_offset_um"] = res_offset
res["decision_checks"] = dict(NAA_neuron_100um=naa_ok, spatial_cross_validation=cv_ok, orientation=orient_ok,
                              residual_offset_um=off_d, effective_band_um=eff_band)
if naa_ok and cv_ok and orient_ok:
    verdict = (f"PASS (global): correct orientation; NAA-neuron positive control and spatial cross-validation passed. "
               f"Cross-modal concordance appears only at {eff_band} µm and coarser -> neighbourhoods of >= 200 µm are the unit of analysis; "
               f"metabolites are not interpreted at single-cell level (per-cell metabolite values of unified_spe represent the neighbourhood)")
    if np.isfinite(off_d) and off_d > 100:
        verdict += f"; residual offset of about {off_d:.0f} µm, refined locally in M2"
elif naa_ok and orient_ok:
    verdict = "WARN: correct orientation and positive control, but cross-validation not passed in all folds -> >= 200-µm bins downstream, re-registration in M2"
else:
    verdict = "FAIL: insufficient spatial agreement of independent markers; re-registration on IF / morphology images required"
hg = tg[tg.metabolite == "2-hydroxyglutarate"]
hg_v = str(hg.verdict.iloc[0]) if len(hg) else ""
res["decision"] = dict(registration=verdict, two_HG=(
    "No adduct of 2-HG in the processed S-BIAD1426 peak list: H3 cannot be tested directly in these data "
    "(the negative-mode [M-H]- ion of 2-HG, m/z 147.0299, would require raw or negative-mode data)")
    if hg_v.startswith("not detected") else hg_v)
C.save_json(res, os.path.join(P["TAB"], "sbiad1426_registration_qc.json"))
rows = []
for k, v in res["scales"].items():
    for h in v.get("held_out", []):
        rows.append(dict(scale=k, test="spatial CV " + h["direction"], rho=h["rho_test"], p_block=h["p_block_perm"]))
    for c in v.get("prespecified_controls", []):
        rows.append(dict(scale=k, test=f"{c['metabolite']} vs {c['reference']}", rho=c["rho"], p_block=c["p_block_perm"], p_naive=c["p_naive"]))
    rows.append(dict(scale=k, test="orientation identity / best alternative", rho=v["orientation_r"]["identity"],
                     note=str(max(vv for kk, vv in v["orientation_r"].items() if kk != "identity"))))
    if "WEHI_control_GPC_vs_OPC_like" in v:
        w = v["WEHI_control_GPC_vs_OPC_like"]
        rows.append(dict(scale=k, test="GPC vs OPC-like (WEHI control)", rho=w["rho"], p_block=w["p_block"]))
        rows.append(dict(scale=k, test="GPC vs OPC-like | WM/GM-adjusted", rho=w["rho_adj_anatomy"], p_block=w["p_block_adj"]))
pd.DataFrame(rows).to_csv(os.path.join(P["TAB"], "sbiad1426_registration_qc_summary.tsv"), sep="\t", index=False)
C.log(f"decision: {verdict}")
C.log(f"2-HG: {res['decision']['two_HG']}")
