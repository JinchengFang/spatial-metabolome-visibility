"""Main Figures 1-6."""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Ellipse, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import sub_kit as K
from sub_kit import OKABE, DESIGN_COL, DESIGN_LAB, REGION_COL
from sub_data import short
import sub_maps as M
import sub_img as I
from fig_common import _cbar, DESIGN_ORDER, variant_name, _mouse, _legend, SN, STR, DA_IONS, SYN

HE_ALPHA = 0.55          # opacity of the grey-scale H&E underlay
GENE_CMAP = LinearSegmentedColormap.from_list("gene", ["#EDEDED", "#A1DAB4", "#41B6C4", "#225EA8", "#081D58"])
SPOT_UM = 85.0           # displayed Visium spot diameter (spots are 55 µm, 100 µm apart)


def _img(F, p, x, y, w, h, bg=None):
    a = F.ax(p, x, y, w, h)
    if bg:
        a.set_facecolor(bg)
    return a


def _rowlabel(F, p, x, y, s, **kw):
    kw.setdefault("fontsize", 6); kw.setdefault("rotation", 90); kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    return F.text(p, x, y, s, **kw)


def _cb(F, p, x, y, w, im, label, ticks=None, ticklabels=None):
    cb = _cbar(F, p, x, y, w, 1.2, im, label)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    return cb


def _cell_map(a, box, d, size=20, alpha=0.95):
    g = I.map4(d["lineage"])
    for lin in ("Other", "Oligodendrocyte", "Malignant", "Neuron"):
        m = g == lin
        if m.any():
            I.dots(a, box, d["cell_xy"][m], None, size, None, 0, 1, color=I.MAP4[lin], alpha=alpha)


def _thumb_crop(img, thr=0.8, pad=6, keep=0.05):
    """Crop a thumbnail to the tissue: bounding box of dark connected components with area >= keep x the largest (drops slide debris and marks)."""
    from scipy import ndimage as ndi
    dark = ndi.binary_opening(img[..., :3].mean(2) < thr, iterations=2)
    lab, n = ndi.label(dark)
    if n == 0:
        return img
    size = np.bincount(lab.ravel())[1:]
    ok = np.isin(lab, np.where(size >= keep * size.max())[0] + 1)
    rows = np.where(ok.any(1))[0]; cols = np.where(ok.any(0))[0]
    out = img[max(0, rows.min() - pad):rows.max() + pad, max(0, cols.min() - pad):cols.max() + pad].copy()
    sub = ok[max(0, rows.min() - pad):rows.max() + pad, max(0, cols.min() - pad):cols.max() + pad]
    out[~ndi.binary_dilation(sub, iterations=8)] = 1.0      # white outside the tissue
    return out


# =============================================================================================== Figure 1
def figure1(P, T, LED, SD, out):
    C, L = M.setup(P)
    S = T.tsv("Step24/m25_sections.tsv"); H = T.tsv("Step24/m25_headline_table.tsv"); DP = T.tsv("Step24/m25_design_pairs.tsv")
    FE = pd.concat([T.tsv("Step23/m24_oracle_features.tsv"), T.tsv("Step24/m24_oracle_features.tsv")], ignore_index=True)
    F = K.Fig("Figure1", 226)
    # ---- example feature: among high-reliability features of same-section DHB b1, the one whose ratio is closest to the section median
    D = M.load_section(P, "v11l12-038-b1"); Sx = D["S"]
    Z = C.zcols(D["Y"]); ZA, ZB = C.zcols(D["YA"]), C.zcols(D["YB"]); det = (np.expm1(D["Y"]) > 0).mean(0); keep = np.where(det >= 0.10)[0]
    _, SB = C.split_half_reliability(ZA[:, keep], ZB[:, keep]); folds = C.spatial_folds(D["xy"], 1000, seed=200)
    Yh = C.cv_ridge_gcv(D["X"], Z[:, keep], folds); r2 = C.r2_columns(Z[:, keep], Yh); rel = np.where(np.nan_to_num(SB) >= 0.7)[0]
    ratio = r2[rel] / SB[rel]; med = np.median(ratio)
    hi_rel = np.where(SB[rel] >= np.percentile(SB[rel], 75))[0]
    j = rel[hi_rel[np.argmin(np.abs(ratio[hi_rel] - med))]]
    obs, pred = Z[:, keep[j]], Yh[:, j]; res = obs - pred; mzj = D["mz"][keep[j]]
    LED.put("fig1c_mz", mzj, "{:.4f}", "Fig1d example feature (DHB b1)"); LED.put("fig1c_R2", r2[j], "{:.2f}"); LED.put("fig1c_SB", SB[j], "{:.2f}")
    LED.put("fig1c_section_median_ratio", med, "{:.2f}")
    box = I.tissue_box(Sx["tx_xy"], 180, square=True)
    # ---------------------------------------------------------------- a three measurements on one section
    F.region("a", 0, 0, 122, 58)
    ion = I.px_ion(Sx, [mzj], ppm=3.0); gene = I.spot_score(Sx, ["MBP"])
    for x_, t_ in ((3, "H&E"), (43, f"MALDI, m/z {mzj:.2f}"), (83, "Visium, $\\it{Mbp}$")):
        F.text("a", x_ + 18.5, 4.0, t_, fontsize=6, ha="center", va="center")
    a1 = _img(F, "a", 3, 6, 37, 37); I.show_he(a1, Sx, box)
    I.scalebar(a1, box, 1.0)
    a2 = _img(F, "a", 43, 6, 37, 37); I.show_he(a2, Sx, box, alpha=HE_ALPHA, grey=True)
    lo, hi = I.robust(ion); im2 = I.squares(a2, box, Sx["px_xy"], ion, I.pitch(Sx["px_xy"]), "magma", lo, hi, alpha=0.95)
    a3 = _img(F, "a", 83, 6, 37, 37); I.show_he(a3, Sx, box, alpha=HE_ALPHA, grey=True)
    lo3, hi3 = I.robust(gene); im3 = I.dots(a3, box, Sx["tx_xy"], gene, SPOT_UM, GENE_CMAP, lo3, hi3)
    _cb(F, "a", 49, 47, 25, im2, "Ion intensity (z)", ticks=[lo, hi], ticklabels=["low", "high"])
    _cb(F, "a", 89, 47, 25, im3, "Expression (z)", ticks=[lo3, hi3], ticklabels=["low", "high"])
    SD.add("Fig1a_MALDI", pd.DataFrame(dict(x_um=Sx["px_xy"][:, 0], y_um=Sx["px_xy"][:, 1], ion_z=ion)))
    SD.add("Fig1a_Visium", pd.DataFrame(dict(x_um=Sx["tx_xy"][:, 0], y_um=Sx["tx_xy"][:, 1], Mbp_z=gene)))
    # ---------------------------------------------------------------- b calibration
    F.region("b", 126, 0, 54, 58)
    b = F.ax("b", 128, 5, 51, 51); K.mm_axes(b, 51, 51)
    K.box(b, 0.5, 38, 24, 10, "Split-half\nreliability  $SB$", fc="#FFFFFF", ec="#7F7F7F", fs=5.5)
    K.box(b, 26.5, 38, 24, 10, "Cross-validated\nprediction  $R^2_{tx}$", fc="#FFFFFF", ec="#7F7F7F", fs=5.5)
    K.box(b, 6, 20, 39, 10, "Positive-control simulation\nattainable  $A$", fc="#FFFFFF", ec="#7F7F7F", fs=5.5)
    K.box(b, 1, 0.5, 49, 11, "Calibrated fraction\n$(R^2_{tx}\\,/\\,SB)\\,/\\,A$", fc="#EAF2FA", ec=OKABE["blue"], fs=6)
    K.arrow(b, 12.5, 38, 18, 30.6); K.arrow(b, 38.5, 38, 33, 30.6); K.arrow(b, 25.5, 20, 25.5, 12.1)
    # ---------------------------------------------------------------- c five designs
    F.region("c", 0, 61, 180, 44)
    tiles = [("same-section", "Same section", "mouse brain · 9 sections"), ("adjacent-section (same mouse, artificial)", "Adjacent pairing", "same mouse · 6 pairs"),
             ("external (another lab)", "External laboratory", "hippocampus · 2 sections"), ("adjacent-section", "Adjacent sections", "ccRCC 5 · GBM 1"),
             ("registered-cells", "Cell-registered", "IDH-mutant glioma")]
    thumbs = []
    D_a1 = M.load_section(P, "v11l12-038-a1")["S"]
    hepath = os.path.join(P["PUB"], "Li2026_COSIE_mouse_hippocampus_MALDI_Visium", "extracted", "Mouse_hippocampus_dataset", "Section1", "Visium", "spatial", "tissue_lowres_image.png")
    idh = I.idh_data(P)
    for k, (des, t1, t2) in enumerate(tiles):
        x0 = 2 + k * 35.8
        t = _img(F, "c", x0 + 3, 66, 29, 26)
        if des == "same-section":
            I.show_he(t, Sx, box)
        elif des.startswith("adjacent-section (same"):
            I.show_he(t, D_a1, I.tissue_box(D_a1["tx_xy"], 180, square=True))
        elif des == "external (another lab)":
            if os.path.exists(hepath):
                im_ = I.white_balance(plt.imread(hepath)[..., :3]); t.imshow(_thumb_crop(im_), interpolation="bilinear", rasterized=True)
            t.set_aspect("equal"); K.noaxis(t)
        elif des == "adjacent-section":
            Sc = I.ccrcc_he(P, "R114_T")
            if Sc is not None:
                I.show_he(t, Sc, I.tissue_box(Sc["tx_xy"], 200, square=True))
            else:
                K.noaxis(t)
        else:
            if idh is not None:
                bx = I.tissue_box(idh["cell_xy"], 150, square=True); I.frame(t, bx, bg="black"); _cell_map(t, bx, idh, size=28)
            else:
                K.noaxis(t)
        t.set_aspect("equal", adjustable="box")
        F.fig.add_artist(Rectangle(((x0 + 3) / F.W, 1 - 94.2 / F.H), 29 / F.W, 1.1 / F.H, transform=F.fig.transFigure, fc=DESIGN_COL[des], ec="none"))
        F.text("c", x0 + 17.5, 97.6, t1, fontsize=6, ha="center", va="center", fontweight="bold")
        F.text("c", x0 + 17.5, 101.5, t2, fontsize=5.5, ha="center", va="center", color="#444444")
        thumbs.append(dict(design=DESIGN_LAB[des], label=f"{t1}; {t2}",
                           image={"same-section": "Vicari SMA V11L12-038 B1 H&E", "adjacent-section (same mouse, artificial)": "Vicari SMA V11L12-038 A1 H&E",
                                  "external (another lab)": "COSIE Section1 Visium low-resolution H&E", "adjacent-section": "SpatialMETA ccRCC R114 Visium H&E",
                                  "registered-cells": "S-BIAD1426 Xenium cell centroids by lineage"}[des]))
    SD.add("Fig1c", pd.DataFrame(thumbs))
    # ---------------------------------------------------------------- d example feature: measured / transcriptome prediction / residual
    F.region("d", 0, 108, 118, 52)
    vmax = float(np.nanpercentile(np.abs(np.r_[obs, pred]), 98))
    labs = ["Measured", "Predicted", "Residual"]
    for k, (v, tt) in enumerate(zip((obs, pred, res), labs)):
        ax_ = _img(F, "d", 3 + k * 38.5, 113, 36, 36); I.show_he(ax_, Sx, box, alpha=0.35, grey=True)
        im = I.squares(ax_, box, D["xy"], v, 200, "RdBu_r", -vmax, vmax); I.corner_label(ax_, tt)
        if k == 0:
            I.scalebar(ax_, box, 1.0)
    _cb(F, "d", 46, 153, 30, im, f"m/z {mzj:.2f}, z-score (200-µm bins)", ticks=[-round(vmax), 0, round(vmax)])
    SD.add("Fig1d", pd.DataFrame(dict(x_um=D["xy"][:, 0], y_um=D["xy"][:, 1], measured_z=obs, predicted_z=pred, residual=res)))
    # ---------------------------------------------------------------- e observed ratio vs attainable ceiling
    F.region("e", 122, 108, 58, 52)
    e = F.ax("e", 133, 112, 44, 30)
    for des in DESIGN_ORDER:
        g = S[S.design == des]
        e.scatter(g.attainable_median, g.ratio_median, s=12, c=DESIGN_COL[des], edgecolor="white", lw=0.3, zorder=3, marker="D" if "artificial" in des else "o")
    xs = np.linspace(0.4, 1.0, 50)
    for c_ in (0.25, 0.5, 0.75, 1.0):
        e.plot(xs, c_ * xs, c="#BBBBBB", lw=0.5, ls="-" if c_ == 1 else "--", zorder=1)
        e.text(0.995, c_ * 0.995 + 0.012, f"{c_:g}", fontsize=5, color="#777777", ha="right", va="bottom")
    e.set_xlim(0.4, 1.0); e.set_ylim(-0.02, 1.0); e.set_xlabel("Attainable fraction $A$"); e.set_ylabel("Observed $R^2_{tx}$ / $SB$")
    hd = [Line2D([], [], marker="D" if "artificial" in d_ else "o", ls="", ms=3, mfc=DESIGN_COL[d_], mec="white", mew=0.3, label=DESIGN_LAB[d_]) for d_ in DESIGN_ORDER]
    _legend(F, "e", hd, 124, 147.5, ncol=1, fontsize=5, handletextpad=0.2, labelspacing=0.15)
    SD.add("Fig1e", S[["dataset", "sample", "design", "n_bins", "n_reliable", "ratio_median", "attainable_median", "calibrated_median"]])
    # ---------------------------------------------------------------- f calibrated fraction per section
    F.region("f", 0, 163, 98, 63)
    S2 = S.copy(); S2["o"] = S2.design.map({d_: i for i, d_ in enumerate(DESIGN_ORDER)}); S2 = S2.sort_values(["o", "sample"]).reset_index(drop=True)
    f = F.ax("f", 33, 166, 61, 50); n = len(S2); yy = np.arange(n)[::-1]
    for y_, r in zip(yy, S2.itertuples()):
        fv = FE[(FE["sample"] == r.sample) & (FE.res_um == 200)].calibrated.dropna()
        if len(fv):
            q1, q3 = np.percentile(fv, [25, 75]); f.plot([q1, q3], [y_, y_], c=DESIGN_COL[r.design], lw=0.9, alpha=0.55, solid_capstyle="butt")
        f.scatter(r.calibrated_median, y_, s=10, c=DESIGN_COL[r.design], zorder=3, marker="D" if "artificial" in r.design else "o", edgecolor="white", lw=0.3)
    f.set_yticks(yy); f.set_yticklabels([short(s_) for s_ in S2["sample"]], fontsize=5)
    f.axvline(1, c="#999999", lw=0.5); f.axvline(0, c="#CCCCCC", lw=0.4); f.set_xlim(-0.12, 1.05); f.set_ylim(-0.7, n - 0.3); f.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    f.set_xlabel("Calibrated fraction")
    for k in range(1, n):
        if S2.o.iloc[k] != S2.o.iloc[k - 1]:
            f.axhline(yy[k] + 0.5, c="#E6E6E6", lw=0.5)
    f.tick_params(axis="y", length=0, pad=1)
    SD.add("Fig1f", S2[["dataset", "sample", "design", "subject", "tissue", "calibrated_median"]])
    # ---------------------------------------------------------------- g mean per design
    F.region("g", 101, 163, 47, 63)
    gx = F.ax("g", 122, 166, 20, 50)
    grp = [("same-section (mouse brain, Vicari)", "Same section", DESIGN_COL["same-section"]), ("external (mouse hippocampus, another lab)", "External", DESIGN_COL["external (another lab)"]),
           ("registered cells (IDH-mutant glioma)", "Cell-registered", DESIGN_COL["registered-cells"]), ("adjacent-section (human tumours)", "Adjacent", DESIGN_COL["adjacent-section"]),
           ("all real designs", "All", "#000000")]
    hc = H[H.metric == "calibrated_median"].set_index("group")
    for k, (g, lab, col) in enumerate(grp):
        r = hc.loc[g]; y_ = len(grp) - 1 - k
        if np.isfinite(r.ci_lo):
            gx.plot([r.ci_lo, r.ci_hi], [y_, y_], c=col, lw=1.0)
        gx.scatter(r.estimate, y_, s=16, c=col, marker="D", zorder=3)
    gx.set_yticks(range(len(grp))[::-1]); gx.set_yticklabels([f"{lab} ({int(hc.loc[g].n_subjects)})" for g, lab, _ in grp], fontsize=5)
    gx.set_xlim(0, 1); gx.set_xticks([0, 0.5, 1]); gx.axvline(1, c="#999999", lw=0.5); gx.set_ylim(-0.6, len(grp) - 0.4); gx.tick_params(axis="y", length=0, pad=1)
    gx.set_xlabel("Calibrated fraction")
    SD.add("Fig1g", hc.reset_index())
    for g, key in (("all real designs", "all"), ("same-section (mouse brain, Vicari)", "same"), ("external (mouse hippocampus, another lab)", "ext"),
                   ("adjacent-section (human tumours)", "adj"), ("registered cells (IDH-mutant glioma)", "cells")):
        r = hc.loc[g]; LED.put(f"cal_{key}", r.estimate, "{:.2f}", "Step24/m25_headline_table.tsv")
        LED.put(f"cal_{key}_lo", r.ci_lo, "{:.2f}"); LED.put(f"cal_{key}_hi", r.ci_hi, "{:.2f}")
        LED.put(f"n_subj_{key}", int(r.n_subjects), "{:d}"); LED.put(f"n_sec_{key}", int(r.n_sections), "{:d}")
    LED.put("unexplained_all", 1 - hc.loc["all real designs"].estimate, "{:.2f}")
    LED.put("unexplained_all_pct", 100 * (1 - hc.loc["all real designs"].estimate), "{:.0f}")
    LED.put("cal_all_pct", 100 * hc.loc["all real designs"].estimate, "{:.0f}")
    att = H[(H.metric == "attainable_median") & (H.group == "all real designs")].estimate.iloc[0]; LED.put("attainable_all", att, "{:.2f}")
    real = S[~S.design.str.contains("artificial")]
    LED.put("attainable_min_200", real.attainable_median.min(), "{:.2f}"); LED.put("attainable_max_200", real.attainable_median.max(), "{:.2f}")
    LED.put("n_sections_total", len(real), "{:d}"); LED.put("n_datasets", real.dataset.nunique(), "{:d}")
    LED.put("cal_same_min", real[real.design == "same-section"].calibrated_median.min(), "{:.2f}"); LED.put("cal_same_max", real[real.design == "same-section"].calibrated_median.max(), "{:.2f}")
    LED.put("cal_ccrcc_min", real[real.dataset == "ccRCC"].calibrated_median.min(), "{:.2f}"); LED.put("cal_ccrcc_max", real[real.dataset == "ccRCC"].calibrated_median.max(), "{:.2f}")
    LED.put("cal_gbm", real[real.dataset == "GBM"].calibrated_median.iloc[0], "{:.2f}")
    for s_ in ("Section1", "Section2"):
        LED.put(f"cal_ext_{s_}", real[real["sample"] == s_].calibrated_median.iloc[0], "{:.2f}")
    # ---------------------------------------------------------------- h paired comparison within the same mouse
    F.region("h", 150, 163, 30, 63)
    hx = F.ax("h", 160, 166, 18, 50)
    for r in DP.itertuples():
        hx.plot([0, 1], [r.cal_same, r.cal_cross], c=OKABE["grey"], lw=0.6)
        hx.scatter([0], [r.cal_same], s=10, c=DESIGN_COL["same-section"], zorder=3); hx.scatter([1], [r.cal_cross], s=10, c=DESIGN_COL["adjacent-section (same mouse, artificial)"], marker="D", zorder=3)
    hx.set_xticks([0, 1]); hx.set_xticklabels(["Same", "Adjacent"], fontsize=5); hx.set_xlim(-0.4, 1.4); hx.set_ylim(-0.05, 0.6)
    hx.set_ylabel("Calibrated fraction")
    nlow = int((DP["diff"] < 0).sum())
    SD.add("Fig1h", DP)
    LED.put("pairs_lower", nlow, "{:d}"); LED.put("pairs_n", len(DP), "{:d}"); LED.put("pairs_retained_median", DP.retained.median(), "{:.2f}")
    LED.put("pairs_retained_min", DP.retained.min(), "{:.2f}"); LED.put("pairs_retained_max", DP.retained.max(), "{:.2f}")
    return F.save(out["main"], out["panels"])


# =============================================================================================== Figure 2
def figure2(P, T, LED, SD, out):
    C, L = M.setup(P)
    S = T.tsv("Step24/m25_sections.tsv"); S = S[~S.design.str.contains("artificial")].copy()
    CAP = T.tsv("Step23/m24x_capacity.tsv"); GRID = T.tsv("Step23/m24x_registration_grid.tsv"); RED = T.tsv("Step23/m24x_redundancy.tsv")
    CON = T.tsv("Step9/m12_conservation_pairs.tsv"); PHY = T.tsv("Step10/m13_physchem_summary.tsv"); MOD = T.tsv("Step9/m12_modules.tsv"); LOPO = T.tsv("Step9/m12_module_lopo.tsv")
    MEM = T.tsv("Step9/m12_module_members.tsv")
    cols = {"Section1": OKABE["green"], "Section2": "#7FCDBB"}
    F = K.Fig("Figure2", 233)
    # a
    F.region("a", 0, 0, 62, 72)
    S["o"] = S.design.map({d: i for i, d in enumerate(DESIGN_ORDER)}); S = S.sort_values(["o", "sample"]).reset_index(drop=True)
    a = F.ax("a", 27, 9, 32, 52); yy = np.arange(len(S))[::-1]
    for y_, r in zip(yy, S.itertuples()):
        a.plot([r.r_resid_null_median, r.r_resid_null_q95], [y_, y_], c="#BBBBBB", lw=2.2, solid_capstyle="butt")
        a.scatter(r.r_resid_median, y_, s=11, c=DESIGN_COL[r.design], zorder=3, edgecolor="white", lw=0.3)
    a.set_yticks(yy); a.set_yticklabels([short(s_) for s_ in S["sample"]], fontsize=5); a.tick_params(axis="y", length=0, pad=1)
    a.set_xlim(0, 0.85); a.set_xticks([0, 0.4, 0.8]); a.set_ylim(-0.7, len(S) - 0.3)
    a.set_xlabel("Residual split-half $r$")
    _legend(F, "a", [Line2D([], [], c="#BBBBBB", lw=2.2, label="Null"), Line2D([], [], marker="o", ls="", ms=3, c="#555555", label="Observed")], 27, 2.5, ncol=2, fontsize=5)
    SD.add("Fig2a", S[["dataset", "sample", "design", "r_resid_median", "r_resid_null_median", "r_resid_null_q95", "null_q95_ridge", "null_q95_projection"]])
    LED.put("rE_min", S.r_resid_median.min(), "{:.2f}"); LED.put("rE_max", S.r_resid_median.max(), "{:.2f}")
    LED.put("rEnull_q95_max", S.r_resid_null_q95.max(), "{:.2f}"); LED.put("rE_frac_above", (S.r_resid_median > S.r_resid_null_q95).mean() * 100, "{:.0f}")
    # b
    F.region("b", 66, 0, 56, 72)
    b = F.ax("b", 92, 9, 28, 52)
    models = list(dict.fromkeys(CAP.model)); yb = np.arange(len(models))[::-1]
    for k, sec in enumerate(("Section1", "Section2")):
        g = CAP[CAP.section == sec].set_index("model").reindex(models)
        for y_, (m, r) in zip(yb, g.iterrows()):
            b.barh(y_ + (0.17 if k == 0 else -0.17), r.ratio_median, 0.32, color=cols[sec], hatch=None if str(r.declared) == "True" else "//////", edgecolor="white", lw=0.3)
    b.set_yticks(yb); b.set_yticklabels([m.replace("RBF kernel ridge", "Kernel ridge") for m in models], fontsize=5); b.tick_params(axis="y", length=0, pad=1)
    b.axvline(0.75, c="k", ls="--", lw=0.5); b.set_xlim(0, 0.8); b.set_xticks([0, 0.25, 0.5, 0.75]); b.set_ylim(-0.6, len(models) - 0.4)
    b.set_xlabel("$R^2_{tx}$ / $SB$")
    _legend(F, "b", [Patch(fc=cols["Section1"], label="H1"), Patch(fc=cols["Section2"], label="H2"), Patch(fc="white", ec="#555555", hatch="//////", label="Post hoc")],
            74, 2.5, ncol=3, fontsize=5, columnspacing=0.6)
    SD.add("Fig2b", CAP)
    dec = CAP[CAP.declared.astype(str) == "True"]
    LED.put("cap_ridge30_s1", CAP[(CAP.section == "Section1") & (CAP.model == "ridge, 30 PCs")].ratio_median.iloc[0], "{:.2f}")
    LED.put("cap_ridge30_s2", CAP[(CAP.section == "Section2") & (CAP.model == "ridge, 30 PCs")].ratio_median.iloc[0], "{:.2f}")
    LED.put("cap_best_declared", dec.ratio_median.max(), "{:.2f}"); LED.put("cap_best_all", CAP.ratio_median.max(), "{:.2f}")
    # c
    F.region("c", 126, 0, 54, 72)
    vmin, vmax = GRID.ratio_median.min(), GRID.ratio_median.max()
    for k, sec in enumerate(("Section1", "Section2")):
        g = GRID[(GRID.section == sec) & (GRID.rot_deg == 0)].pivot(index="dy_um", columns="dx_um", values="ratio_median").sort_index(ascending=False)
        c = F.ax("c", 136 + k * 22, 8, 19, 19)
        im = c.imshow(g.values, cmap="viridis", vmin=vmin, vmax=vmax, extent=(-250, 250, -250, 250))
        c.scatter([0], [0], marker="+", c="white", s=14, lw=0.7)
        c.set_xticks([-200, 0, 200]); c.set_yticks([-200, 0, 200]); c.tick_params(labelsize=5)
        c.set_title(f"H{k + 1}", fontsize=5.5, loc="center", pad=1.5)
        if k == 0:
            c.set_ylabel("Δy (µm)")
        else:
            c.set_yticklabels([])
        c.set_xlabel("Δx (µm)")
        best = GRID[GRID.section == sec].ratio_median.max(); orig = GRID[(GRID.section == sec) & (GRID.dx_um == 0) & (GRID.dy_um == 0) & (GRID.rot_deg == 0)].ratio_median.iloc[0]
        LED.put(f"reg_orig_{sec}", orig, "{:.2f}"); LED.put(f"reg_best_{sec}", best, "{:.2f}")
    _cb(F, "c", 141, 36.5, 33, im, "$R^2_{tx}$ / $SB$")
    c2 = F.ax("c", 141, 47, 35, 14)
    for k, sec in enumerate(("Section1", "Section2")):
        g = GRID[GRID.section == sec]; best = g.ratio_median.max(); orig = g[(g.dx_um == 0) & (g.dy_um == 0) & (g.rot_deg == 0)].ratio_median.iloc[0]
        c2.plot([0, 1], [orig, best], c=list(cols.values())[k], lw=0.8, marker="o", ms=2.5)
    c2.set_xticks([0, 1]); c2.set_xticklabels(["Registered", "Best of 75"], fontsize=5); c2.set_xlim(-0.3, 1.3)
    c2.set_ylim(0, 0.8); c2.set_yticks([0, 0.4, 0.8]); c2.axhline(0.75, c="k", ls="--", lw=0.5); c2.set_ylabel("$R^2_{tx}$ / $SB$")
    SD.add("Fig2c", GRID)
    # d
    F.region("d", 0, 75, 58, 46)
    d = F.ax("d", 11, 79, 44, 28)
    cats = [("R2_tx", "Transcriptome"), ("R2_maldi_all_others", "Metabolome"), ("R2_maldi_pruned_pixel", "− homologues"), ("R2_maldi_pruned_pixel_and_bin0.8", "− |r| > 0.8")]
    for k, sec in enumerate(("Section1", "Section2")):
        g = RED[RED.section == sec]
        for j_, (cname, _) in enumerate(cats):
            q1, q2, q3 = np.nanpercentile(g[cname], [25, 50, 75]); x = j_ + (k - 0.5) * 0.3
            d.plot([x, x], [q1, q3], c=list(cols.values())[k], lw=1.0); d.scatter(x, q2, s=12, c=list(cols.values())[k], zorder=3, edgecolor="white", lw=0.3)
    d.set_xticks(range(len(cats))); d.set_xticklabels([c_[1] for c_ in cats], fontsize=5, rotation=25, ha="right")
    d.set_ylabel("Cross-validated $R^2$"); d.set_ylim(0, 1); d.set_xlim(-0.5, len(cats) - 0.5); d.axvline(0.5, c="#DDDDDD", lw=0.5)
    SD.add("Fig2d", RED)
    for sec in ("Section1", "Section2"):
        g = RED[RED.section == sec]; LED.put(f"red_tx_{sec}", g.R2_tx.median(), "{:.2f}"); LED.put(f"red_all_{sec}", g.R2_maldi_all_others.median(), "{:.2f}")
        LED.put(f"red_pruned_{sec}", g.R2_maldi_pruned_pixel.median(), "{:.2f}"); LED.put(f"red_strict_{sec}", g["R2_maldi_pruned_pixel_and_bin0.8"].median(), "{:.2f}")
    # e
    F.region("e", 62, 75, 54, 46)
    e = F.ax("e", 76, 79, 36, 30)
    for r in CON.itertuples():
        e.plot([0, 1], [r.rho_raw, r.rho_residual], c="#BBBBBB", lw=0.6)
    e.scatter(np.zeros(len(CON)), CON.rho_raw, s=10, c=OKABE["grey"], zorder=3); e.scatter(np.ones(len(CON)), CON.rho_residual, s=10, c=OKABE["vermillion"], zorder=3)
    e.set_xticks([0, 1]); e.set_xticklabels(["Metabolite", "Residual"], fontsize=5.5); e.set_xlim(-0.4, 1.4)
    e.set_ylim(0, 1); e.set_ylabel("Between-patient Mantel ρ")
    SD.add("Fig2e", CON)
    LED.put("mantel_resid_median", CON.rho_residual.median(), "{:.2f}"); LED.put("mantel_pairs_sig", int((CON.p_residual < 0.05).sum()), "{:d}"); LED.put("mantel_pairs_n", len(CON), "{:d}")
    # f
    F.region("f", 120, 75, 60, 46)
    f = F.ax("f", 150, 79, 26, 30)
    sub = [("all", "All pairs"), ("cross_class", "Cross-class"), ("dmz_gt_200", "Δm/z > 200"), ("dmassdefect_gt_0.15", "Δmass defect > 0.15"),
           ("partial_mantel", "Partial Mantel"), ("cross_polarity", "Opposite polarity")]
    g = PHY[(PHY.group == "ccRCC") & (PHY.kind == "residual")].set_index("subset")
    yf = np.arange(len(sub))[::-1]
    for y_, (s_, lab) in zip(yf, sub):
        r = g.loc[s_]; f.barh(y_, r.rho_median, 0.6, color=OKABE["vermillion"] if s_ != "cross_polarity" else "#F0A57A")
        f.text(r.rho_median + 0.02, y_, f"{100 * r.frac_p05:.0f}%", fontsize=5, va="center")
    f.set_yticks(yf); f.set_yticklabels([l_ for _, l_ in sub], fontsize=5); f.tick_params(axis="y", length=0, pad=1); f.set_xlim(0, 0.9); f.set_ylim(-0.6, len(sub) - 0.4)
    f.set_xlabel("Residual Mantel ρ")
    SD.add("Fig2f", g.reset_index())
    LED.put("phys_crosspol_rho", g.loc["cross_polarity"].rho_median, "{:.2f}"); LED.put("phys_crosspol_frac", 100 * g.loc["cross_polarity"].frac_p05, "{:.0f}")
    LED.put("phys_partial_rho", g.loc["partial_mantel"].rho_median, "{:.2f}"); LED.put("phys_R2", 100 * g.physchem_R2_of_mean_residual_cov.iloc[0], "{:.0f}")
    # g module OM1 in five patients (H&E + module score map)
    F.region("g", 0, 124, 180, 71)
    pats = ["R114_T", "S15_T", "X49_T", "Y27_T", "Y7_T"]
    mod_show = MOD.sort_values("mean_residual_coherence", ascending=False).module.iloc[0]
    mem = MEM[MEM.module == mod_show].feature.astype(str).values
    rows = []; im = None
    _rowlabel(F, "g", 5, 143, "H&E"); _rowlabel(F, "g", 5, 172, mod_show)
    for k, s in enumerate(pats):
        f_ = os.path.join(P["PROC"], "m12_residuals", f"ccRCC_{s}_200um.npz")
        x0 = 9 + k * 33.6
        t1 = _img(F, "g", x0, 129, 31, 28); t2 = _img(F, "g", x0, 158, 31, 28)
        F.text("g", x0 + 15.5, 126.8, s.replace("_T", ""), fontsize=6, ha="center", va="center")
        if not os.path.exists(f_):
            K.noaxis(t1); K.noaxis(t2); continue
        z = np.load(f_, allow_pickle=True); xy = z["xy"]; names = z["names"].astype(str); E = z["E"]
        jx = pd.Index(names).get_indexer(mem); jx = jx[jx >= 0]
        Mz = C.zcols(E[:, jx]); U, S_, Vt = np.linalg.svd(Mz, full_matrices=False); ev = U[:, 0] * S_[0] * np.sign(Vt[0].sum() or 1.0)
        ev = ev / (np.std(ev) + 1e-9)
        Sc = I.ccrcc_he(P, s); bx = I.tissue_box(xy, 250, square=True)
        if Sc is not None:
            I.show_he(t1, Sc, bx); I.show_he(t2, Sc, bx, alpha=0.3, grey=True)
        else:
            I.frame(t1, bx); I.frame(t2, bx)
        im = I.squares(t2, bx, xy, ev, 200, "RdBu_r", -2.5, 2.5)
        if k == 0:
            I.scalebar(t1, bx, 1.0)
        rows.append(pd.DataFrame(dict(patient=s, x_um=xy[:, 0], y_um=xy[:, 1], module_score=ev)))
    if im is not None:
        _cb(F, "g", 76, 188.2, 28, im, f"{mod_show} score (s.d.)", ticks=[-2, 0, 2])
    if rows:
        SD.add("Fig2g", pd.concat(rows, ignore_index=True))
    # h residual consistency within modules
    F.region("h", 0, 198, 92, 35)
    pc = [c_ for c_ in MOD.columns if c_.startswith("coh_")]
    Mx = MOD.set_index("module")[pc]; Mx = Mx.loc[sorted(Mx.index)]
    hx = F.ax("h", 16, 203, 50, 23)
    imh = hx.imshow(Mx.values, cmap="magma", vmin=0, vmax=1, aspect="auto")
    hx.set_yticks(range(len(Mx))); hx.set_yticklabels(list(Mx.index), fontsize=5)
    hx.set_xticks(range(len(pc))); hx.set_xticklabels([p_.replace("coh_", "").replace("_T", "") for p_ in pc], fontsize=5)
    hx.xaxis.tick_top(); hx.tick_params(axis="both", length=0, pad=1)
    for i_ in range(Mx.shape[0]):
        for j_ in range(Mx.shape[1]):
            hx.text(j_, i_, f"{Mx.values[i_, j_]:.2f}", ha="center", va="center", fontsize=5, color="white" if Mx.values[i_, j_] < 0.6 else "black")
    _cbar(F, "h", 70, 215, 18, 1.2, imh, "Coherence")
    SD.add("Fig2h", MOD.sort_values("module")[["module", "size"] + pc])
    LED.put("n_modules", len(MOD), "{:d}"); LED.put("lopo_null_min", LOPO.null_mean.min(), "{:.2f}"); LED.put("lopo_null_max", LOPO.null_mean.max(), "{:.2f}")
    LED.put("fig2g_module", mod_show, "{}")
    # i module composition (chemical class)
    F.region("i", 96, 198, 84, 35)
    ix_ = F.ax("i", 107, 202, 46, 23)
    cls = MEM.copy(); cls["cls"] = cls.lipid_class.fillna("unannotated").astype(str).replace({"nan": "unannotated", "": "unannotated"})
    top = cls.cls.value_counts().index[:6].tolist()
    cls["cls"] = np.where(cls.cls.isin(top), cls.cls, "other")
    order = top + ["other"]; pal = ["#BBBBBB", OKABE["blue"], OKABE["orange"], OKABE["green"], OKABE["vermillion"], OKABE["purple"], "#E3E3E3"]
    tab = pd.crosstab(cls.module, cls.cls).reindex(index=sorted(Mx.index), columns=order).fillna(0)
    frac = tab.div(tab.sum(1), axis=0); left = np.zeros(len(frac))
    yv = np.arange(len(frac))[::-1]
    for k_, cname in enumerate(order):
        ix_.barh(yv, frac[cname].values, 0.7, left=left, color=pal[k_ % len(pal)], ec="white", lw=0.3); left += frac[cname].values
    ix_.set_yticks(yv); ix_.set_yticklabels(list(frac.index), fontsize=5); ix_.set_xlim(0, 1); ix_.set_xticks([0, 0.5, 1]); ix_.set_xlabel("Fraction of members")
    ix_.tick_params(axis="y", length=0, pad=1)
    _legend(F, "i", [Patch(fc=pal[k_ % len(pal)], label=c_) for k_, c_ in enumerate(order)], 155, 202, ncol=1, fontsize=5, labelspacing=0.15, handlelength=0.9)
    SD.add("Fig2i", tab.reset_index())
    return F.save(out["main"], out["panels"])


# =============================================================================================== Figure 3
def _schema_ns(a, w, h):
    """Schematic: substantia nigra (somata, synthesis mRNA) -> striatum (terminals, MSN targets)."""
    K.mm_axes(a, w, h)
    a.add_patch(Ellipse((w * 0.5, h * 0.70), w * 0.86, h * 0.26, fc="#DCEAF6", ec=REGION_COL[STR], lw=0.6))
    a.add_patch(Ellipse((w * 0.5, h * 0.27), w * 0.70, h * 0.20, fc="#FBE3D6", ec=REGION_COL[SN], lw=0.6))
    for k in range(4):
        x0 = w * (0.28 + 0.15 * k); y0 = h * 0.27
        a.add_patch(Ellipse((x0, y0), 3.0, 2.2, fc=REGION_COL[SN], ec="none", zorder=3))
        x1 = w * (0.22 + 0.19 * k); y1 = h * (0.66 + 0.04 * (k % 2))
        a.add_patch(FancyArrowPatch((x0, y0 + 1.1), (x1, y1), connectionstyle=f"arc3,rad={0.15 * (1 if k % 2 else -1)}", arrowstyle="-", lw=0.6, color=REGION_COL[SN]))
        a.scatter([x1], [y1], s=6, c=REGION_COL[SN], zorder=3)
    for (x, y) in ((0.3, 0.76), (0.55, 0.78), (0.75, 0.74), (0.42, 0.62), (0.66, 0.62)):
        a.add_patch(Ellipse((w * x, h * y), 2.2, 1.6, fc="white", ec=REGION_COL[STR], lw=0.5))
    a.text(w * 0.5, h * 0.985, "Striatum", ha="center", va="top", fontsize=6, fontweight="bold", color=REGION_COL[STR])
    a.text(w * 0.5, h * 0.94, "terminals · MSN targets", ha="center", va="top", fontsize=5.3, color="#333333")
    a.text(w * 0.5, h * 0.105, "Substantia nigra", ha="center", va="top", fontsize=6, fontweight="bold", color=REGION_COL[SN])
    a.text(w * 0.5, h * 0.06, "somata · $\\it{Th, Ddc, Slc6a3, Slc18a2}$", ha="center", va="top", fontsize=5.3, color="#333333")


def _section_row(F, p, P, sec, y0, x_list, kind="da", ST=None, label=None, r_show=None):
    """One row of tissue maps: H&E | dopamine (MALDI pixels) | synthesis genes (Visium spots)."""
    D = M.load_section(P, sec, mode="m19"); S_ = D["S"]; box = I.tissue_box(S_["tx_xy"], 180, square=True); out = {}
    a1 = _img(F, p, x_list[0], y0, 34, 34); I.show_he(a1, S_, box)
    da = I.px_ion(S_, DA_IONS, ppm=10.0); lo, hi = I.robust(da)
    a2 = _img(F, p, x_list[1], y0, 34, 34); I.show_he(a2, S_, box, alpha=HE_ALPHA, grey=True)
    out["im_da"] = I.squares(a2, box, S_["px_xy"], da, I.pitch(S_["px_xy"]), "magma", lo, hi, alpha=0.95)
    sy = I.spot_score(S_, SYN); lo2, hi2 = I.robust(sy)
    a3 = _img(F, p, x_list[2], y0, 34, 34); I.show_he(a3, S_, box, alpha=HE_ALPHA, grey=True)
    out["im_syn"] = I.dots(a3, box, S_["tx_xy"], sy, SPOT_UM, GENE_CMAP, lo2, hi2)
    if r_show is not None:
        I.corner_label(a2, f"$r$ = {K.num(r_show)}", color="black", loc="bl")
    out.update(D=D, S=S_, box=box, axes=(a1, a2, a3), da=da, syn=sy, lims=((lo, hi), (lo2, hi2)))
    return out


def figure3(P, T, LED, SD, out):
    C, L = M.setup(P)
    MS = T.tsv("Step13/m18_molecule_section.tsv"); ST = T.tsv("Step17/m20_strict_targeted.tsv"); RB = T.tsv("Step14/m18_robustness_summary.tsv")
    F = K.Fig("Figure3", 160)
    F.region("a", 0, 0, 48, 92)
    a = F.ax("a", 2, 6, 44, 84); _schema_ns(a, 44, 84)
    F.region("b", 52, 0, 128, 92)
    xs = [64, 102, 140]
    for k, t_ in enumerate(("H&E", "Dopamine", "Synthesis genes")):
        F.text("b", xs[k] + 17, 3.6, t_, fontsize=6, ha="center", va="center")
    rows = {}
    for row, (sec, area, lab) in enumerate((("v11t17-085_a1", SN, "SN"), ("v11l12-109-a1", STR, "STR"))):
        rt = ST[(ST.molecule == "dopamine") & (ST["sample"].astype(str).str.contains(sec.split("-", 1)[1]))]
        rv = rows[lab] = _section_row(F, "b", P, sec, 6 + row * 38, xs, r_show=float(rt.r.iloc[0]) if len(rt) else None)
        _rowlabel(F, "b", 58.5, 23 + row * 38, lab, color=REGION_COL[area], fontweight="bold")
        # consistency check with the strict-test table: recompute r on 200-µm bins
        D = rv["D"]; y = M.zmean(D["Y"][:, M.ion_index(D["mz"], DA_IONS)]); s, _ = M.gene_score(D, SYN)
        rv["r_bins"] = float(np.corrcoef(y, s)[0, 1])
        if len(rt):
            assert abs(float(rt.r.iloc[0]) - rv["r_bins"]) < 0.01, (sec, rv["r_bins"])
        SD.add(f"Fig3b_{lab}_MALDI", pd.DataFrame(dict(x_um=rv["S"]["px_xy"][:, 0], y_um=rv["S"]["px_xy"][:, 1], dopamine_z=rv["da"])))
        SD.add(f"Fig3b_{lab}_Visium", pd.DataFrame(dict(x_um=rv["S"]["tx_xy"][:, 0], y_um=rv["S"]["tx_xy"][:, 1], synthesis_z=rv["syn"])))
        if row == 0:
            I.scalebar(rv["axes"][0], rv["box"], 1.0)
    (lo, hi), (lo2, hi2) = rows["STR"]["lims"]
    _cb(F, "b", 106, 86, 26, rows["STR"]["im_da"], "Dopamine (z)", ticks=[lo, hi], ticklabels=["low", "high"])
    _cb(F, "b", 144, 86, 26, rows["STR"]["im_syn"], "Synthesis score (z)", ticks=[lo2, hi2], ticklabels=["low", "high"])
    # c
    F.region("c", 0, 97, 58, 63)
    c = F.ax("c", 12, 102, 43, 44)
    xsx = {SN: 0, STR: 1}
    for mol, dx, mk in (("dopamine", -0.18, "o"), ("serotonin", 0.18, "s")):
        g = ST[ST.molecule == mol]
        for r in g.itertuples():
            x = xsx[r.brain_area] + dx + np.random.default_rng(abs(hash(r.sample)) % 2**32).uniform(-0.05, 0.05)
            c.plot([x, x], [r.null_median_dual, r.r], c="#CCCCCC", lw=0.5, zorder=1)
            c.scatter(x, r.r, s=13, marker=mk, c=REGION_COL[r.brain_area] if mol == "dopamine" else "white", edgecolor=REGION_COL[r.brain_area], lw=0.6, zorder=3)
            c.scatter(x, r.null_median_dual, s=5, c="#999999", marker="_", zorder=2)
    c.axhline(0, c="#999999", lw=0.4); c.set_xticks([0, 1]); c.set_xticklabels(["SN", "STR"], fontsize=6); c.set_xlim(-0.5, 1.5)
    c.set_ylabel("$r$ with own synthesis genes"); c.set_ylim(-0.12, 0.5)
    dsn = ST[(ST.molecule == "dopamine") & (ST.brain_area == SN)]; dst = ST[(ST.molecule == "dopamine") & (ST.brain_area == STR)]
    c.text(0, 0.47, f"{int((dsn.p_dual_matched < 0.05).sum())}/{len(dsn)}", ha="center", fontsize=5.5); c.text(1, 0.47, f"{int((dst.p_dual_matched < 0.05).sum())}/{len(dst)}", ha="center", fontsize=5.5)
    _legend(F, "c", [Line2D([], [], marker="o", ls="", ms=3, mfc="#555555", mec="#555555", label="Dopamine"), Line2D([], [], marker="s", ls="", ms=3, mfc="white", mec="#555555", label="Serotonin"),
                     Line2D([], [], marker="_", ls="", ms=4, mec="#999999", label="Null")], 4, 154, ncol=3, fontsize=5, columnspacing=0.5)
    SD.add("Fig3c", ST[ST.molecule.isin(["dopamine", "serotonin"])])
    LED.put("da_r_SN_min", dsn.r.min(), "{:.2f}"); LED.put("da_r_SN_max", dsn.r.max(), "{:.2f}"); LED.put("da_r_STR_min", dst.r.min(), "{:.2f}"); LED.put("da_r_STR_max", dst.r.max(), "{:.2f}")
    LED.put("da_SN_sig", int((dsn.p_dual_matched < 0.05).sum()), "{:d}"); LED.put("da_STR_sig", int((dst.p_dual_matched < 0.05).sum()), "{:d}")
    LED.put("da_SN_geneset_moran_pctl", dsn.geneset_moran_pctl.median(), "{:.3f}")
    # d
    F.region("d", 62, 97, 56, 63)
    d = F.ax("d", 74, 102, 41, 44)
    g = MS[MS.molecule == "dopamine"]
    for k, area in enumerate((SN, STR)):
        e_ = g[g.brain_area == area]
        for j_, (col, lab, fc) in enumerate((("R2_synthesis_only", "Synthesis", OKABE["orange"]), ("R2_target_cells_only", "MSN", OKABE["sky"]))):
            x = k + (j_ - 0.5) * 0.36
            d.bar(x, e_[col].median(), 0.32, color=fc)
            d.scatter(np.full(len(e_), x), e_[col], s=5, c="k", zorder=3)
    d.axhline(0, c="k", lw=0.4); d.set_xticks([0, 1]); d.set_xticklabels(["SN", "STR"], fontsize=6); d.set_xlim(-0.6, 1.6)
    d.set_ylabel("Cross-validated $R^2$, dopamine"); d.set_ylim(-0.1, 0.17)
    _legend(F, "d", [Patch(fc=OKABE["orange"], label="Synthesis genes"), Patch(fc=OKABE["sky"], label="MSN genes")], 66, 154, ncol=2, fontsize=5)
    SD.add("Fig3d", g[["sample", "brain_area", "mouse", "R2_synthesis_only", "R2_target_cells_only", "rho_synthesis", "p_synthesis"]])
    for area, k_ in ((SN, "SN"), (STR, "STR")):
        e_ = g[g.brain_area == area]; LED.put(f"da_R2syn_{k_}", e_.R2_synthesis_only.median(), "{:.3f}"); LED.put(f"da_R2tgt_{k_}", e_.R2_target_cells_only.median(), "{:.3f}")
    # e
    F.region("e", 122, 97, 58, 63)
    e = F.ax("e", 132, 102, 45, 40)
    RB2 = RB.reset_index(drop=True); xv = np.arange(len(RB2))
    for x_, r in zip(xv, RB2.itertuples()):
        e.plot([x_, x_], [r.rho_STR, r.rho_SN], c="#CCCCCC", lw=0.5, zorder=1)
    e.scatter(xv, RB2.rho_SN, s=9, c=REGION_COL[SN], zorder=3); e.scatter(xv, RB2.rho_STR, s=9, c=REGION_COL[STR], zorder=3)
    e.set_xticks(xv); e.set_xticklabels([variant_name(v) for v in RB2.variant], rotation=90, fontsize=5)
    e.axhline(0, c="#999999", lw=0.4); e.set_ylabel("Median $r$"); e.set_ylim(-0.1, 0.55); e.set_xlim(-0.7, len(RB2) - 0.3)
    e.tick_params(axis="x", length=0, pad=1)
    _legend(F, "e", [Line2D([], [], marker="o", ls="", ms=3, c=REGION_COL[SN], label="SN"), Line2D([], [], marker="o", ls="", ms=3, c=REGION_COL[STR], label="STR")],
            140, 99, ncol=2, fontsize=5)
    SD.add("Fig3e", RB2)
    LED.put("rob_n_variants", len(RB2), "{:d}"); LED.put("rob_frac_contrast", 100 * (RB2.rho_SN > RB2.rho_STR).mean(), "{:.0f}")
    LED.put("rob_frac_D54", 100 * RB2.D54.astype(str).eq("True").mean(), "{:.0f}"); LED.put("rob_frac_D55", 100 * RB2.D55.astype(str).eq("True").mean(), "{:.0f}")
    LED.put("da_r_map_SN_mPD1", rows["SN"]["r_bins"], "{:.2f}"); LED.put("da_r_map_STR_mPD1", rows["STR"]["r_bins"], "{:.2f}")
    return F.save(out["main"], out["panels"])


# =============================================================================================== Figure 4
def figure4(P, T, LED, SD, out):
    R = T.tsv("Step20/m22r_sections.tsv"); V = T.json("Step20/m22r_verdict.json"); SEN = T.tsv("Step20/m22r_sensitivity.tsv")
    F = K.Fig("Figure4", 160)
    R2 = R[(R["shift"].astype(str) == "True") & (R.res_um == 200)]
    # a schematic
    F.region("a", 0, 0, 48, 90)
    a = F.ax("a", 2, 8, 44, 70); K.mm_axes(a, 44, 70)
    a.add_patch(Ellipse((22, 38), 40, 46, fc="#F4F4F4", ec="#888888", lw=0.6)); a.plot([22, 22], [14, 62], c="k", ls="--", lw=0.6)
    a.text(11, 64, "Lesioned", ha="center", va="bottom", fontsize=6, color=OKABE["vermillion"]); a.text(33, 64, "Intact", ha="center", va="bottom", fontsize=6, color=OKABE["blue"])
    for (x, y) in ((12, 46), (15, 36), (10, 28), (14, 22)):
        a.add_patch(Rectangle((x - 1.5, y - 1.5), 3, 3, fc=OKABE["vermillion"], ec="none")); a.add_patch(Rectangle((44 - x - 1.5, y - 1.5), 3, 3, fc=OKABE["blue"], ec="none"))
        a.add_patch(FancyArrowPatch((x + 1.7, y), (44 - x - 1.7, y), arrowstyle="<->", mutation_scale=4, lw=0.4, color="#555555"))
    a.text(22, 8, "Δ = intact − lesioned", ha="center", fontsize=6)
    a.text(22, 2, "ΔDA ≈ $b$ · Δsynthesis", ha="center", fontsize=6)
    # b tissue maps of mPD3 SN and STR: H&E + lesion side | dopamine | synthesis genes
    F.region("b", 52, 0, 128, 92)
    xs = [64, 102, 140]
    for k, t_ in enumerate(("H&E · lesion side", "Dopamine", "Synthesis genes")):
        F.text("b", xs[k] + 17, 3.6, t_, fontsize=6, ha="center", va="center")
    keep = {}
    for row, (sec, lab, area) in enumerate((("v11t17-085_b1", "SN", SN), ("v11l12-109-b1", "STR", STR))):
        rv = _section_row(F, "b", P, sec, 6 + row * 38, xs)
        D = M.load_section(P, sec); S_ = D["S"]; lab_ = S_.get("tx_labels", {}).get("lesion")
        if lab_ is not None:
            ls = pd.Series(np.asarray(lab_).astype(str)).str.lower()
            col = np.where(ls.str.startswith("lesion"), OKABE["vermillion"], np.where(ls.str.startswith("intact"), OKABE["blue"], "#00000000"))
            m = col != "#00000000"
            I.dots(rv["axes"][0], rv["box"], S_["tx_xy"][m], None, SPOT_UM, None, 0, 1, color=col[m], alpha=0.30)
        _rowlabel(F, "b", 58.5, 23 + row * 38, lab, color=REGION_COL[area], fontweight="bold")
        keep[lab] = rv
        SD.add(f"Fig4b_{lab}_MALDI", pd.DataFrame(dict(x_um=rv["S"]["px_xy"][:, 0], y_um=rv["S"]["px_xy"][:, 1], dopamine_z=rv["da"])))
        SD.add(f"Fig4b_{lab}_Visium", pd.DataFrame(dict(x_um=S_["tx_xy"][:, 0], y_um=S_["tx_xy"][:, 1], synthesis_z=rv["syn"],
                                                         hemisphere=np.asarray(lab_).astype(str) if lab_ is not None else "NA")))
        if row == 0:
            I.scalebar(rv["axes"][0], rv["box"], 1.0)
    (lo, hi), (lo2, hi2) = keep["STR"]["lims"]
    _cb(F, "b", 106, 86, 26, keep["STR"]["im_da"], "Dopamine (z)", ticks=[lo, hi], ticklabels=["low", "high"])
    _cb(F, "b", 144, 86, 26, keep["STR"]["im_syn"], "Synthesis score (z)", ticks=[lo2, hi2], ticklabels=["low", "high"])
    _legend(F, "b", [Patch(fc=OKABE["vermillion"], alpha=0.6, label="Lesioned"), Patch(fc=OKABE["blue"], alpha=0.6, label="Intact")], 64, 83.5, ncol=2, fontsize=5)
    # c
    F.region("c", 0, 96, 58, 64)
    c = F.ax("c", 13, 101, 42, 44)
    da = R2[R2.molecule == "dopamine"].copy(); da["mouse"] = da["sample"].map(_mouse)
    for k, area in enumerate((SN, STR)):
        e_ = da[da.brain_area == area].sort_values("mouse")
        for j_, (col, fc) in enumerate((("mean_dY", OKABE["black"]), ("mean_dSyn", OKABE["orange"]))):
            x = k + (j_ - 0.5) * 0.36
            c.bar(x, e_[col].mean(), 0.32, color="white", ec=fc, lw=0.8)
            c.scatter(x + np.linspace(-0.08, 0.08, len(e_)), e_[col], s=7, c=fc, zorder=3)
    c.axhline(0, c="k", lw=0.4); c.set_xticks([0, 1]); c.set_xticklabels(["SN", "STR"], fontsize=6); c.set_xlim(-0.6, 1.6)
    c.set_ylabel("Mean Δ (s.d.)")
    _legend(F, "c", [Line2D([], [], marker="o", ls="", ms=3, c=OKABE["black"], label="Dopamine"), Line2D([], [], marker="o", ls="", ms=3, c=OKABE["orange"], label="Synthesis score")],
            4, 153, ncol=2, fontsize=5)
    SD.add("Fig4c", da)
    # d
    F.region("d", 62, 96, 58, 64)
    d = F.ax("d", 76, 101, 41, 44)
    rows = []
    for k, (area, key) in enumerate(((SN, "SN_animals"), (STR, "STR_animals"))):
        for j_, r in enumerate(sorted(V[key], key=lambda z: z["mouse"])):
            x = k + (j_ - 1) * 0.22
            d.plot([x, x], [r["ci_lo"], r["ci_hi"]], c=REGION_COL[area], lw=0.9); d.scatter(x, r["frac_explained"], s=12, c=REGION_COL[area], zorder=3)
            d.text(x, max(r["ci_hi"], r["frac_explained"]) + 0.04, r["mouse"].replace("mPD", "m"), ha="center", fontsize=5)
            rows.append(dict(area=area, **r))
    d.axhline(0, c="k", lw=0.4); d.set_xticks([0, 1]); d.set_xticklabels(["SN", "STR"], fontsize=6); d.set_xlim(-0.6, 1.6); d.set_ylim(-0.35, 0.95)
    d.set_ylabel("Explained fraction")
    SD.add("Fig4d", pd.DataFrame(rows))
    for r in V["SN_animals"]:
        LED.put(f"frac_SN_{r['mouse']}", r["frac_explained"], "{:.2f}"); LED.put(f"frac_SN_{r['mouse']}_lo", r["ci_lo"], "{:.2f}"); LED.put(f"frac_SN_{r['mouse']}_hi", r["ci_hi"], "{:.2f}")
        LED.put(f"p_SN_{r['mouse']}", r["p_frac"], "{:.3f}")
    for r in V["STR_animals"]:
        LED.put(f"frac_STR_{r['mouse']}", r["frac_explained"], "{:.3f}"); LED.put(f"p_STR_{r['mouse']}", r["p_frac"], "{:.2f}")
    LED.put("SN_stouffer_p", V["SN_stouffer_p"], "{:.2f}")
    for r in da.itertuples():
        LED.put(f"dDA_{'SN' if r.brain_area == SN else 'STR'}_{r.mouse}", r.mean_dY, "{:.2f}"); LED.put(f"dSyn_{'SN' if r.brain_area == SN else 'STR'}_{r.mouse}", r.mean_dSyn, "{:.2f}")
    # e
    F.region("e", 124, 96, 56, 64)
    e = F.ax("e", 136, 101, 41, 44)
    lab = [f"{'shift' if str(r.shift) == 'True' else 'none'}\n{int(r.res_um)} µm" for r in SEN.itertuples()]
    xv = np.arange(len(SEN))
    for x_, r in zip(xv, SEN.itertuples()):
        e.plot([x_, x_], [r.STR_median, r.SN_median], c="#CCCCCC", lw=0.5, zorder=1)
    e.scatter(xv, SEN.SN_median, s=11, c=REGION_COL[SN], zorder=3); e.scatter(xv, SEN.STR_median, s=11, c=REGION_COL[STR], zorder=3)
    e.set_xticks(xv); e.set_xticklabels(lab, fontsize=5); e.axhline(0, c="k", lw=0.4); e.set_ylim(-0.05, 0.35); e.set_xlim(-0.5, len(SEN) - 0.5)
    e.set_ylabel("Median explained fraction"); e.tick_params(axis="x", length=0, pad=1)
    _legend(F, "e", [Line2D([], [], marker="o", ls="", ms=3, c=REGION_COL[SN], label="SN"), Line2D([], [], marker="o", ls="", ms=3, c=REGION_COL[STR], label="STR")],
            140, 98.5, ncol=2, fontsize=5)
    SD.add("Fig4e", SEN)
    LED.put("sens_holds", int(SEN.contrast_holds.astype(str).eq("True").sum()), "{:d}"); LED.put("sens_n", len(SEN), "{:d}")
    return F.save(out["main"], out["panels"])


# =============================================================================================== Figure 5
def figure5(P, T, LED, SD, out):
    C, L = M.setup(P)
    NA = T.tsv("Step15/m19c_mouse_naa.tsv"); ST = T.tsv("Step17/m20_strict_targeted.tsv"); GI = T.tsv("Step15/m19c_glioma_interaction.tsv"); CS = T.tsv("Step15/m19c_cptac_sensitivity.tsv")
    NEURON = ["NAT8L"] + C.PROGRAMS["Neuron_cortex"].split(); OLIGO = ["ASPA"] + C.PROGRAMS["WM_oligodendrocyte"].split()
    F = K.Fig("Figure5", 230)
    # a schematic
    F.region("a", 0, 0, 40, 50)
    a = F.ax("a", 1, 6, 38, 42); K.mm_axes(a, 38, 42)
    K.box(a, 2, 25, 34, 14, "Neuron\n$\\it{NAT8L}$\nAsp + AcCoA → NAA", fc="#FBE3D6", ec=OKABE["vermillion"], fs=5.5)
    K.box(a, 2, 1, 34, 14, "Oligodendrocyte\n$\\it{ASPA}$\nNAA → Asp + acetate", fc="#DCEAF6", ec=OKABE["blue"], fs=5.5)
    K.arrow(a, 19, 25, 19, 15.4)
    # b mouse: H&E | NAA | neuronal side | oligodendroglial side
    F.region("b", 43, 0, 137, 50)
    D = M.load_section(P, "v11l12-038-b1", mode="m19"); S_ = D["S"]; box = I.tissue_box(S_["tx_xy"], 180, square=True)
    ions = [float(x) for x in str(NA[NA["sample"] == "v11l12-038-b1_DHB"].naa_ions.iloc[0]).split(",")]
    naa = I.px_ion(S_, ions, ppm=10.0); sn_s = I.spot_score(S_, NEURON); so_s = I.spot_score(S_, OLIGO)
    xs = [46, 79, 112, 145]
    tt = ["H&E", "NAA", "Neuronal side", "Oligodendroglial side"]
    ims = []
    for k in range(4):
        ax_ = _img(F, "b", xs[k], 7, 31, 31)
        F.text("b", xs[k] + 15.5, 4.0, tt[k], fontsize=6, ha="center", va="center")
        if k == 0:
            I.show_he(ax_, S_, box); I.scalebar(ax_, box, 1.0)
        else:
            I.show_he(ax_, S_, box, alpha=HE_ALPHA, grey=True)
            if k == 1:
                lo, hi = I.robust(naa); ims.append(I.squares(ax_, box, S_["px_xy"], naa, I.pitch(S_["px_xy"]), "magma", lo, hi, alpha=0.95))
            else:
                v = sn_s if k == 2 else so_s; lo, hi = I.robust(v); ims.append(I.dots(ax_, box, S_["tx_xy"], v, SPOT_UM, GENE_CMAP, lo, hi))
    y_ = M.zmean(D["Y"][:, M.ion_index(D["mz"], ions)]); sn, _ = M.gene_score(D, NEURON); so, _ = M.gene_score(D, OLIGO)
    rn, ro = float(np.corrcoef(y_, sn)[0, 1]), float(np.corrcoef(y_, so)[0, 1])
    I.corner_label(F.fig.axes[-2], f"$r$ = {K.num(rn)}", loc="bl"); I.corner_label(F.fig.axes[-1], f"$r$ = {K.num(ro)}", loc="bl")
    _cb(F, "b", 83, 41.5, 24, ims[0], "NAA (z)", ticks=list(ims[0].get_clim()), ticklabels=["low", "high"])
    _cb(F, "b", 132, 41.5, 24, ims[2], "Gene score (z)", ticks=list(ims[2].get_clim()), ticklabels=["low", "high"])
    SD.add("Fig5b_MALDI", pd.DataFrame(dict(x_um=S_["px_xy"][:, 0], y_um=S_["px_xy"][:, 1], NAA_z=naa)))
    SD.add("Fig5b_Visium", pd.DataFrame(dict(x_um=S_["tx_xy"][:, 0], y_um=S_["tx_xy"][:, 1], neuronal_z=sn_s, oligodendroglial_z=so_s)))
    # c neuronal vs oligodendroglial side
    F.region("c", 0, 54, 44, 74)
    c = F.ax("c", 11, 60, 31, 46)
    for k, r in enumerate(NA.itertuples()):
        c.plot([k - 0.15, k + 0.15], [r.neuronal_r, r.oligodendroglial_r], c="#CCCCCC", lw=0.5)
        c.scatter(k - 0.15, r.neuronal_r, s=13, c=OKABE["vermillion"], zorder=3); c.scatter(k + 0.15, r.oligodendroglial_r, s=13, c=OKABE["blue"], zorder=3)
    c.set_xticks(range(len(NA))); c.set_xticklabels([short(s_).replace("STR ", "").replace(" · ", "\n") for s_ in NA["sample"]], fontsize=5); c.axhline(0, c="k", lw=0.4)
    c.set_ylabel("$r$ with NAA"); c.set_ylim(-0.25, 0.6); c.set_xlim(-0.6, len(NA) - 0.4)
    _legend(F, "c", [Line2D([], [], marker="o", ls="", ms=3, c=OKABE["vermillion"], label="Neuronal"), Line2D([], [], marker="o", ls="", ms=3, c=OKABE["blue"], label="Oligodendroglial")],
            3, 117, ncol=1, fontsize=5)
    SD.add("Fig5c", NA.merge(ST[ST.molecule == "N-acetylaspartate"][["sample", "p_expr_matched", "p_dual_matched", "null_median_dual"]], on="sample", how="left"))
    LED.put("naa_rn_min", NA.neuronal_r.min(), "{:.2f}"); LED.put("naa_rn_max", NA.neuronal_r.max(), "{:.2f}"); LED.put("naa_ro_min", NA.oligodendroglial_r.min(), "{:.2f}")
    LED.put("naa_ro_max", NA.oligodendroglial_r.max(), "{:.2f}"); LED.put("naa_rank_min", 100 * (1 - NA.NAA_rank_pct_neuronal.min()), "{:.1f}")
    sd = ST[ST.molecule == "N-acetylaspartate"]; LED.put("naa_dual_sig", int((sd.p_dual_matched < 0.05).sum()), "{:d}"); LED.put("naa_dual_n", len(sd), "{:d}")
    LED.put("naa_map_rn", rn, "{:.2f}"); LED.put("naa_map_ro", ro, "{:.2f}")
    # d IDH-mutant glioma: Xenium cell types | NAA (MALDI) | zoom
    F.region("d", 48, 54, 132, 74)
    d = I.idh_data(P)
    if d is not None:
        bx = I.tissue_box(d["cell_xy"], 120, square=True)
        d1 = _img(F, "d", 50, 60, 41, 41); I.frame(d1, bx, bg="black"); _cell_map(d1, bx, d, size=22); I.scalebar(d1, bx, 1.0, color="white")
        d2 = _img(F, "d", 94, 60, 41, 41); I.frame(d2, bx, bg="black")
        img = I.binned(d2, bx, d["px_xy"], d["naa"], 50, None, 0, 1, sigma_bins=1.0, min_n=3, return_img=True); lo, hi = np.nanpercentile(img, [2, 99])
        imn = I.binned(d2, bx, d["px_xy"], d["naa"], 50, "magma", lo, hi, sigma_bins=1.0, min_n=3)
        # zoom window: 1.8 mm square at the position with most neurons (weighted by malignant cells)
        neu = d["cell_xy"][d["lineage"] == "Neuron"]; W_ = 1800.0
        gx_ = np.arange(bx[0], bx[1] - W_, 200); gy_ = np.arange(bx[2], bx[3] - W_, 200); best, zb = -1, None
        for x0 in gx_:
            for y0 in gy_:
                n_ = ((neu[:, 0] >= x0) & (neu[:, 0] < x0 + W_) & (neu[:, 1] >= y0) & (neu[:, 1] < y0 + W_)).sum()
                mal = ((d["cell_xy"][:, 0] >= x0) & (d["cell_xy"][:, 0] < x0 + W_) & (d["cell_xy"][:, 1] >= y0) & (d["cell_xy"][:, 1] < y0 + W_) & (d["lineage"] == "Malignant")).sum()
                sc_ = n_ * np.sqrt(max(mal, 1))
                if sc_ > best:
                    best, zb = sc_, (x0, x0 + W_, y0, y0 + W_)
        for a_ in (d1, d2):
            a_.add_patch(Rectangle((zb[0], zb[2]), W_, W_, fill=False, ec="white", lw=0.6, zorder=8))
        d3 = _img(F, "d", 138, 60, 41, 41); I.frame(d3, zb, bg="black")
        I.binned(d3, zb, d["px_xy"], d["naa"], 30, "Greys_r", lo, hi, sigma_bins=1.0, min_n=2)
        g4 = I.map4(d["lineage"])
        for lin, s_ in (("Malignant", 16), ("Oligodendrocyte", 16), ("Neuron", 22)):
            m = g4 == lin; I.dots(d3, zb, d["cell_xy"][m], None, s_, None, 0, 1, color=I.MAP4[lin], alpha=0.95)
        I.scalebar(d3, zb, 0.5, color="white")
        for x_, t_ in ((50, "Xenium cell types"), (94, "NAA (MALDI)"), (138, "Inset")):
            F.text("d", x_ + 20.5, 57.6, t_, fontsize=6, ha="center", va="center")
        _cb(F, "d", 101, 104, 26, imn, "NAA (z)", ticks=[lo, hi], ticklabels=["low", "high"])
        _legend(F, "d", [Line2D([], [], marker="o", ls="", ms=3, c=I.MAP4[l_], label=l_) for l_ in ("Malignant", "Neuron", "Oligodendrocyte", "Other")],
                50, 104, ncol=2, fontsize=5, columnspacing=0.6, handletextpad=0.1)
        SD.add("Fig5d_cells", pd.DataFrame(dict(x_um=d["cell_xy"][:, 0], y_um=d["cell_xy"][:, 1], lineage=d["lineage"], annotation=d["anno"])))
        bimg = I.binned(None, bx, d["px_xy"], d["naa"], 50, None, 0, 1, sigma_bins=0.0, min_n=3, return_img=True)
        yy_, xx_ = np.where(np.isfinite(bimg))
        SD.add("Fig5d_NAA_50um", pd.DataFrame(dict(x_um=bx[0] + (xx_ + 0.5) * 50, y_um=bx[2] + (yy_ + 0.5) * 50, NAA_z=bimg[yy_, xx_])))
    # e IDH β
    F.region("e", 0, 132, 88, 48)
    e = F.ax("e", 16, 136, 46, 34)
    g = GI[GI.kind == "IDHmut_cells"].sort_values("res_um"); xv = np.arange(len(g))
    e.errorbar(xv - 0.12, g.beta_neuron, yerr=1.96 * g.null_sd, fmt="o", ms=3, c=OKABE["vermillion"], lw=0.8, capsize=1.5)
    e.errorbar(xv + 0.12, g.beta_interaction, yerr=1.96 * g.null_sd, fmt="s", ms=3, c=OKABE["grey"], lw=0.8, capsize=1.5)
    e.axhline(0, c="k", lw=0.4); e.set_xticks(xv); e.set_xticklabels([f"{int(r)} µm" for r in g.res_um], fontsize=5.5); e.set_ylabel("Standardized β")
    e.set_xlim(-0.5, len(g) - 0.5); e.set_ylim(-0.3, 1.0)
    _legend(F, "e", [Line2D([], [], marker="o", ls="", ms=3, c=OKABE["vermillion"], label="Neurons"), Line2D([], [], marker="s", ls="", ms=3, c=OKABE["grey"], label="Neurons × malignant")],
            64, 140, ncol=1, fontsize=5)
    SD.add("Fig5e", g)
    g2 = g[g.res_um == 200].iloc[0]
    LED.put("idh_beta_neuron", g2.beta_neuron, "{:.2f}"); LED.put("idh_p_neuron", g2.p_neuron_pos, "{:.3f}"); LED.put("idh_beta_int", g2.beta_interaction, "{:.2f}"); LED.put("idh_p_int", g2.p_int_neg, "{:.2f}")
    # f GBM
    F.region("f", 92, 132, 88, 48)
    f = F.ax("f", 108, 136, 46, 34)
    gb = GI[GI.kind == "GBM_programs"].copy(); gb["ion"] = gb.metabolite.str.extract(r"\[(.*)\]")[0]; pats = sorted(gb["sample"].unique())
    for k, p_ in enumerate(pats):
        for j_, ion in enumerate(sorted(gb.ion.unique())):
            r = gb[(gb["sample"] == p_) & (gb.ion == ion)]
            if len(r):
                r = r.iloc[0]; x = k + (j_ - 0.5) * 0.3
                f.scatter(x, r.beta_neuron, s=12, marker="o" if j_ == 0 else "^", c=OKABE["vermillion"] if r.p_neuron_pos < 0.05 else "white", edgecolor=OKABE["vermillion"], lw=0.6, zorder=3)
    f.axhline(0, c="k", lw=0.4); f.set_xticks(range(len(pats)))
    f.set_xticklabels([p_.replace("UKF", "") + ("*" if gb[gb["sample"] == p_].identity_high.astype(str).eq("True").any() else "") for p_ in pats], fontsize=5.5)
    f.set_xlabel("GBM patient"); f.set_ylabel("β, NAA ~ neuronal"); f.set_xlim(-0.6, len(pats) - 0.4)
    _legend(F, "f", [Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec=OKABE["vermillion"], label="[M−H]$^{-}$"), Line2D([], [], marker="^", ls="", ms=3, mfc="white", mec=OKABE["vermillion"], label="[M+Cl]$^{-}$"),
                     Line2D([], [], marker="o", ls="", ms=3, mfc=OKABE["vermillion"], mec=OKABE["vermillion"], label="$P$ < 0.05")], 156, 140, ncol=1, fontsize=5)
    SD.add("Fig5f", gb)
    LED.put("gbm_pos", int((gb.beta_neuron > 0).sum()), "{:d}"); LED.put("gbm_n", len(gb), "{:d}"); LED.put("gbm_sig", int((gb.p_neuron_pos < 0.05).sum()), "{:d}")
    # g CPTAC model variants
    VN = {"base": "Base", "neuron_linear": "Neuronal, linear", "neuron_cubic": "Neuronal, cubic", "mature_neuron_genes": "Mature-neuron genes",
          "rank": "Rank-transformed", "plus_astro_myeloid_vascular": "+ astro, myeloid, vascular"}
    F.region("g", 0, 184, 88, 46)
    gq = F.ax("g", 33, 188, 50, 28)
    var = [v for v in CS.variant.unique() if not v.startswith("program:")]
    for k, v in enumerate(var):
        for j_, (tg, fc, mk) in enumerate((("NAA", OKABE["vermillion"], "o"), ("NAT8L", OKABE["grey"], "s"))):
            r = CS[(CS.variant == v) & (CS.target == tg)].iloc[0]; y0 = len(var) - 1 - k + (0.15 if j_ == 0 else -0.15)
            gq.scatter(r.beta_malignant, y0, s=11, marker=mk, c=fc if r.p_neg < 0.05 else "white", edgecolor=fc, lw=0.6, zorder=3)
    gq.axvline(0, c="k", lw=0.4); gq.set_yticks(range(len(var))[::-1]); gq.set_yticklabels([VN.get(v, v) for v in var], fontsize=5); gq.tick_params(axis="y", length=0, pad=1)
    gq.set_xlabel("β, malignant programme (CPTAC)"); gq.set_xlim(-0.4, 0.1); gq.set_ylim(-0.6, len(var) - 0.4)
    _legend(F, "g", [Line2D([], [], marker="o", ls="", ms=3, c=OKABE["vermillion"], label="NAA"), Line2D([], [], marker="s", ls="", ms=3, c=OKABE["grey"], label="$\\it{NAT8L}$"),
                     Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="#555555", label="$P$ ≥ 0.05")], 4, 224.5, ncol=3, fontsize=5)
    SD.add("Fig5g", CS[~CS.variant.str.startswith("program:")])
    nb = CS[(CS.variant == "base") & (CS.target == "NAA")].iloc[0]; LED.put("cptac_naa_beta", nb.beta_malignant, "{:.2f}"); LED.put("cptac_naa_p", nb.p_neg, "{:.3f}")
    nv = CS[(~CS.variant.str.startswith("program:")) & (CS.target == "NAA")]; LED.put("cptac_naa_robust", int((nv.p_neg < 0.05).sum()), "{:d}"); LED.put("cptac_naa_nvar", len(nv), "{:d}")
    LED.put("cptac_naa_beta_min", nv.beta_malignant.min(), "{:.2f}"); LED.put("cptac_naa_beta_max", nv.beta_malignant.max(), "{:.2f}")
    # h malignant state programmes
    F.region("h", 92, 184, 88, 46)
    hq = F.ax("h", 122, 188, 54, 28)
    prog = [v for v in CS.variant.unique() if v.startswith("program:")]
    for k, v in enumerate(prog):
        for j_, (tg, fc, mk) in enumerate((("NAA", OKABE["vermillion"], "o"), ("NAT8L", OKABE["grey"], "s"))):
            r = CS[(CS.variant == v) & (CS.target == tg)].iloc[0]; y0 = len(prog) - 1 - k + (0.15 if j_ == 0 else -0.15)
            hq.scatter(r.beta_malignant, y0, s=11, marker=mk, c=fc if r.p_neg < 0.05 else "white", edgecolor=fc, lw=0.6, zorder=3)
    hq.axvline(0, c="k", lw=0.4); hq.set_yticks(range(len(prog))[::-1])
    hq.set_yticklabels([{"AC_like": "AC-like", "OPC_like": "OPC-like", "NPC_like": "NPC-like", "MES1_inflammatory": "MES1", "Hypoxia_MES2": "MES2 (hypoxia)", "Cycling": "Cycling"}.get(v.replace("program:", ""), v)
                        for v in prog], fontsize=5)
    hq.tick_params(axis="y", length=0, pad=1); hq.set_xlabel("β, state programme (CPTAC)"); hq.set_ylim(-0.6, len(prog) - 0.4); hq.set_xlim(-0.45, 0.45)
    SD.add("Fig5h", CS[CS.variant.str.startswith("program:")])
    for v in prog:
        for tg in ("NAA", "NAT8L"):
            r = CS[(CS.variant == v) & (CS.target == tg)].iloc[0]; k_ = v.replace("program:", "")
            LED.put(f"prog_{tg}_{k_}_beta", r.beta_malignant, "{:.2f}"); LED.put(f"prog_{tg}_{k_}_p", r.p_neg, "{:.3f}")
    res = F.save(out["main"], out["panels"])
    tab = NA[NA["sample"] == "v11l12-038-b1_DHB"].iloc[0]
    for lab_, mine, ref in (("neuronal", rn, tab.neuronal_r), ("oligodendroglial", ro, tab.oligodendroglial_r)):
        if abs(mine - ref) > 0.03:
            res["issues"].append(f"value check failed: NAA {lab_} r from map {mine:.3f} vs result table {ref:.3f}"); res["n_issues"] += 1
    if d is None:
        res["issues"].append("IDH-mutant Xenium / MALDI processed data missing (02_processed/sbiad1426_*.h5ad): panel 5d is empty"); res["n_issues"] += 1
    return res


# =============================================================================================== Figure 6
def figure6(P, T, LED, SD, out):
    SP = T.tsv("Step16/m20_sma_specificity.tsv"); SC = T.tsv("Step17/m20_spec_curve.tsv"); GS = T.tsv("Step7/m9_gem_summary.tsv")
    PW = T.tsv("Step18/m21_power_grid.tsv"); VS = T.json("Step17/m20_strict_verdict.json"); V16 = T.json("Step16/m20_verdict.json"); VC = T.json("Step17/m20_cptac_verdict.json")
    VSC = T.json("Step17/m20_spec_verdict.json")
    F = K.Fig("Figure6", 118)
    F.region("a", 0, 0, 58, 56)
    a = F.ax("a", 12, 5, 43, 37)
    g = SP[SP.level == "full"]
    a.hist(g.pctl, bins=np.linspace(0, 1, 21), color=OKABE["blue"], edgecolor="white", lw=0.3)
    a.axhline(len(g) / 20, c="k", ls="--", lw=0.5); a.set_xlabel("Own-pathway percentile"); a.set_ylabel("Tests")
    SD.add("Fig6a", g)
    LED.put("gem_n_tests", VS["gem"]["n_tests"], "{:d}"); LED.put("gem_n_metab", VS["gem"]["n_metabolites"], "{:d}"); LED.put("gem_frac_sig", 100 * VS["gem"]["frac_sig"], "{:.0f}")
    LED.put("gem_binom_p", VS["gem"]["binom_p"], "{:.3f}"); LED.put("gem_frac_sig_detr", 100 * V16["mouse_detrended"]["frac_sig"], "{:.0f}")
    F.region("b", 61, 0, 55, 56)
    b = F.ax("b", 72, 5, 41, 37)
    items = []
    for mol, area, lab in (("dopamine", SN, "DA, SN"), ("dopamine", STR, "DA, STR"), ("N-acetylaspartate", STR, "NAA")):
        e_ = g[(g.metabolite == mol) & (g.brain_area == area)]; items.append((lab, e_.pctl.values))
    for k, (lab, v) in enumerate(items):
        b.bar(k, np.mean(v), 0.6, color=[REGION_COL[SN], REGION_COL[STR], OKABE["purple"]][k], alpha=0.85)   # bar = mean, as reported in the text (gem_da_*)
        b.scatter(np.full(len(v), k), v, s=6, c="k", zorder=3)
    b.axhline(0.5, c="k", ls="--", lw=0.5); b.axhline(0.95, c="#999999", ls=":", lw=0.5); b.set_ylim(0, 1.05)
    b.set_xticks(range(len(items))); b.set_xticklabels([i[0] for i in items], fontsize=5.5); b.set_ylabel("Own-pathway percentile")
    SD.add("Fig6b", g[g.metabolite.isin(["dopamine", "N-acetylaspartate"])])
    LED.put("gem_da_SN", V16["dopamine_by_area"]["Substantia Nigra"], "{:.2f}"); LED.put("gem_da_STR", V16["dopamine_by_area"]["Striatum"], "{:.2f}")
    for (lab, v), key in zip(items[:2], ("Substantia Nigra", "Striatum")):   # bar heights must equal the reported numbers
        assert abs(np.mean(v) - V16["dopamine_by_area"][key]) < 0.005, (lab, np.mean(v), V16["dopamine_by_area"][key])
    F.region("c", 119, 0, 61, 56)
    c = F.ax("c", 141, 5, 36, 37)
    gs = GS[(GS.level == "full") & (GS.res_um == 200)].copy()
    labs = [short(s_) for s_ in gs["sample"]] + ["CPTAC GBM"]; vals = list(100 * gs.frac_p05_gem) + [100 * VC["full"]["frac_sig"]]; ns = list(gs.n.astype(int)) + [VC["full"]["n"]]
    yv = np.arange(len(labs))[::-1]
    c.barh(yv, vals, 0.6, color=[OKABE["vermillion"]] * len(gs) + [OKABE["grey"]])
    for y_, v_, n_ in zip(yv, vals, ns):
        c.text(59, y_, f"{n_}", fontsize=5, va="center", ha="right")
    c.text(59, len(labs) - 0.35, "$n$", fontsize=5, va="center", ha="right")
    c.axvline(5, c="k", ls="--", lw=0.5); c.set_yticks(yv); c.set_yticklabels(labs, fontsize=5); c.tick_params(axis="y", length=0, pad=1)
    c.set_xlabel("Metabolites with specificity (%)"); c.set_xlim(0, 60); c.set_ylim(-0.6, len(labs) - 0.1)
    SD.add("Fig6c", pd.concat([gs, pd.DataFrame([dict(sample="CPTAC", n=VC["full"]["n"], frac_p05_gem=VC["full"]["frac_sig"])])], ignore_index=True))
    LED.put("cptac_gem_frac", 100 * VC["full"]["frac_sig"], "{:.0f}"); LED.put("cptac_gem_n", VC["full"]["n"], "{:d}")
    F.region("d", 0, 60, 88, 58)
    d = F.ax("d", 12, 64, 73, 34)
    sc = SC[SC.level == "full"].sort_values("rho").reset_index(drop=True)
    d.scatter(np.arange(len(sc)), sc.rho, s=3, c=np.where(sc.p_one_sided < 0.05, OKABE["blue"], "#BBBBBB"), lw=0)
    d.axhline(0, c="k", lw=0.4); d.set_xlabel("Specification (ranked)"); d.set_ylabel("ρ, export share vs specificity")
    d.set_xlim(-2, len(sc) + 1)
    sub = VSC["full"]["by_subset"]
    _legend(F, "d", [Line2D([], [], marker="o", ls="", ms=2.5, c=OKABE["blue"], label="$P$ < 0.05"), Line2D([], [], marker="o", ls="", ms=2.5, c="#BBBBBB", label="$P$ ≥ 0.05")],
            14, 64.5, ncol=2, fontsize=5)
    SD.add("Fig6d", sc)
    LED.put("spec_n", len(sc), "{:d}"); LED.put("spec_frac_neg", 100 * VSC["full"]["frac_negative"], "{:.0f}"); LED.put("spec_base_rho", VSC["base_rho"], "{:.2f}")
    LED.put("spec_ci_lo", VSC["base_boot_ci95"][0], "{:.2f}"); LED.put("spec_ci_hi", VSC["base_boot_ci95"][1], "{:.3f}"); LED.put("spec_noDHB", sub["drop_DHB"], "{:.2f}")
    LED.put("spec_noAA", sub["no_amino_acids"], "{:.2f}"); LED.put("cptac_export_rho", VC["full"]["rho"], "{:.2f}")
    F.region("e", 92, 60, 88, 58)
    e = F.ax("e", 104, 64, 72, 34)
    pw = PW[(PW.hypothesis == "H2_hypoxia_lowers_NAA_per_neuron") & (PW.res_um == 200) & (PW.tau == 0.10) & (PW.alpha == 0.025) & (PW.delta == 0.15)]
    for des, ls in (("same_section", "-"), ("adjacent_section", "--")):
        for k_, col in ((2, OKABE["vermillion"]), (4, OKABE["blue"])):
            g_ = pw[(pw.design == des) & (pw.regions_per_patient == k_)].sort_values("n_patients")
            e.plot(g_.n_patients, g_.power, ls=ls, c=col, lw=0.9, marker="o", ms=2)
    e.axhline(0.8, c="#999999", lw=0.5, ls=":"); e.set_xlabel("Patients"); e.set_ylabel("Power"); e.set_ylim(0, 1.02)
    _legend(F, "e", [Line2D([], [], c="k", ls="-", label="Same section"), Line2D([], [], c="k", ls="--", label="Adjacent"),
                     Line2D([], [], c=OKABE["vermillion"], label="2 regions"), Line2D([], [], c=OKABE["blue"], label="4 regions")], 130, 82, ncol=2, fontsize=5, columnspacing=0.6)
    SD.add("Fig6e", pw)
    rec = T.json("Step18/m21_recommendation.json")["H2_hypoxia_lowers_NAA_per_neuron"]["n_for_80pct_power"]
    LED.put("pow_same_k2", rec["same_section_200um_k2"], "{:d}"); LED.put("pow_adj_k2", rec["adjacent_section_200um_k2"], "{:d}"); LED.put("pow_same_k4", rec["same_section_200um_k4"], "{:d}")
    r1 = T.json("Step18/m21_recommendation.json"); LED.put("pow_H1", r1["H1_NAA_tracks_neurons"]["base_case"], "{:d}"); LED.put("pow_H3", r1["H3_interaction"]["base_case"], "{:d}")
    return F.save(out["main"], out["panels"])
