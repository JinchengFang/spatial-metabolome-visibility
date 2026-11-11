"""Supplementary Figures 1-10."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sub_kit as K
from sub_kit import OKABE, DESIGN_COL, DESIGN_LAB, REGION_COL
from sub_data import short
from fig_common import _cbar, DESIGN_ORDER, _legend, SN, STR, variant_name



def _tbl(a, w, h, header, rows, xs, fs=5, top_pad=2.0, row_h=3.4):
    K.mm_axes(a, w, h)
    for x_, t_ in zip(xs, header):
        a.text(x_, h - top_pad, t_, fontsize=fs, fontweight="bold", va="center")
    a.plot([0, w], [h - top_pad - 1.8, h - top_pad - 1.8], c="k", lw=0.4)
    for i, r in enumerate(rows):
        y_ = h - top_pad - 4.0 - i * row_h
        for x_, t_ in zip(xs, r):
            a.text(x_, y_, str(t_), fontsize=fs, va="center")
    a.plot([0, w], [h - top_pad - 4.0 - (len(rows) - 0.5) * row_h] * 2, c="k", lw=0.4)


# ------------------------------------------------------------------------------------------- S1 data and registration
def supp1(P, T, LED, SD, out):
    S = T.tsv("Step24/m25_sections.tsv"); RG = T.tsv("Step10/m13_registration.tsv"); RV = T.tsv("Step2/ravi_registration.tsv"); ID = T.tsv("Step3/ravi_identity_final.tsv")
    IM = T.tsv("Step10/m13_imzml_summary.tsv"); X = T.tsv("Step22/m23x_summary.tsv")
    F = K.Fig("SupplementaryFigure1", 126, kind="supp")
    F.region("a", 0, 0, 180, 42)
    a = F.ax("a", 3, 4, 177, 33)
    rows = []
    for ds, g in S[~S.design.str.contains("artificial")].groupby("dataset", sort=False):
        plat = {"IDH-mut": "Xenium + MALDI (pos.)", "GBM": "Visium + MALDI (neg.)", "ccRCC": "Visium + MALDI (pos./neg.)", "SMA-mouse": "Visium + MALDI (DHB, 9-AA, FMP-10)",
                "External-COSIE": "Visium + MALDI"}.get(ds, "")
        src = {"IDH-mut": "BioImage Archive S-BIAD1426", "GBM": "Ravi et al. 2022 (Dryad)", "ccRCC": "SpatialMETA (Zenodo 14986870)", "SMA-mouse": "Vicari et al. 2024",
               "External-COSIE": "COSIE (Zenodo 22948067)"}.get(ds, "")
        des = DESIGN_LAB.get(g.design.iloc[0], g.design.iloc[0])
        rows.append([ds.replace("External-COSIE", "Mouse hippocampus").replace("SMA-mouse", "Mouse brain").replace("IDH-mut", "IDH-mutant glioma"), plat, des, src,
                     g.subject.nunique(), len(g), int(g.n_bins.median()), int(g.n_reliable.sum())])
    _tbl(a, 177, 33, ["Dataset", "Platforms", "Design", "Source", "Subjects", "Sections", "Bins*", "Reliable†"], rows, [0, 23, 57, 94, 131, 143, 155, 166], row_h=5.5)
    SD.add("SFig1a", pd.DataFrame(rows, columns=["dataset", "platforms", "design", "source", "subjects", "sections", "median_bins_200um", "reliable_features_total"]))
    # b SMA registration: outline IoU and molecular check
    F.region("b", 0, 44, 90, 82)
    b = F.ax("b", 30, 48, 56, 56)
    RG = RG.reset_index(drop=True); yb = np.arange(len(RG))[::-1]
    b.barh(yb + 0.18, RG.iou, 0.34, color=OKABE["blue"], label="Outline IoU")
    b.barh(yb - 0.18, RG.diag_pc_r_best, 0.34, color=OKABE["sky"], label="Molecular check, chosen")
    b.scatter(RG.diag_pc_r_rot180, yb - 0.18, marker="|", c="k", s=20, lw=0.8, label="Molecular check, 180° rotated", zorder=3)
    lab = {r.section: f"{r.section.replace('v11l12-', '').replace('v11t17-', '')} ({IM.set_index('section').loc[r.section, 'matrix']}, {IM.set_index('section').loc[r.section, 'mouse']})" for r in RG.itertuples()}
    b.set_yticks(yb); b.set_yticklabels([lab[s_] for s_ in RG.section], fontsize=5); b.tick_params(axis="y", length=0, pad=1); b.set_xlim(0, 1)
    b.set_xlabel("Registration score"); b.set_ylim(-0.6, len(RG) - 0.4)
    _legend(F, "b", [Patch(fc=OKABE["blue"], label="Outline IoU"), Patch(fc=OKABE["sky"], label="Molecular check (r), chosen"),
                     Line2D([], [], marker="|", ls="", c="k", ms=5, label="Molecular check, 180°")], 4, 118, ncol=2, fontsize=5)
    SD.add("SFig1b", RG)
    LED.put("sma_iou_min", RG.iou.min(), "{:.2f}"); LED.put("sma_iou_max", RG.iou.max(), "{:.2f}")
    # c human GBM registration and identity
    F.region("c", 94, 44, 86, 44)
    c = F.ax("c", 112, 48, 64, 26)
    RV = RV.merge(ID[["section", "patient", "identity_confidence"]].rename(columns={"patient": "molecular_patient"}), on="section", how="left")
    yc = np.arange(len(RV))[::-1]
    c.barh(yc, RV.iou, 0.6, color=[OKABE["vermillion"] if str(p_) == "UKF260" else "#F0A57A" for p_ in RV.molecular_patient])
    c.set_yticks(yc); c.set_yticklabels([f"MALDI {r.section} → {r.molecular_patient}" for r in RV.itertuples()], fontsize=5); c.tick_params(axis="y", length=0, pad=1)
    c.set_xlim(0, 1); c.set_xlabel("Outline IoU (GBM, Ravi et al.)")
    SD.add("SFig1c", RV)
    # d registration of the external data
    F.region("d", 94, 90, 86, 36)
    d = F.ax("d", 112, 94, 64, 12)
    x2 = X[X.res_um == 200].reset_index(drop=True)
    d.barh([1, 0], x2.reg_score, 0.45, color=OKABE["green"]); d.set_ylim(-0.6, 1.6)
    d.set_yticks([1, 0]); d.set_yticklabels(["Hippocampus 1", "Hippocampus 2"], fontsize=5); d.set_xlim(0, 1.05); d.set_xlabel("Shape overlap after similarity transform")
    SD.add("SFig1d", x2)
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S2 noise ceilings
def supp2(P, T, LED, SD, out):
    CS = T.tsv("Step7/m9_ceiling_summary.tsv"); g = CS[(CS.res_um == 200) & (CS.level == "full")].reset_index(drop=True)
    F = K.Fig("SupplementaryFigure2", 74, kind="supp")
    labs = [short(s_) for s_ in g["sample"]]; x = np.arange(len(g))
    specs = [("a", "SB", "Split-half reliability (Spearman–Brown)", (0, 1)), ("b", None, "Fraction of reliable variance", (0, 1.05)),
             ("c", "depth_gain", "$R^2_{tx}$ gain, full vs half depth", (-0.01, 0.06)), ("d", "shared_pixel_noise", "Shared pixel-noise contribution", (0, 0.06))]
    for k, (p, col, yl, lim) in enumerate(specs):
        F.region(p, k * 45, 0, 45, 74); a = F.ax(p, k * 45 + 11.5, 4, 31.5, 46)      # 11.5 mm left margin for tick labels and axis title
        if col:
            a.bar(x, g[col], 0.65, color=OKABE["grey"])
        else:
            a.bar(x - 0.2, g.tx_norm_reliable, 0.38, color=OKABE["vermillion"], label="Transcriptome")
            a.bar(x + 0.2, g.maldi_cross_norm, 0.38, color=OKABE["blue"], label="Other metabolites (cross-half)")
            _legend(F, p, [Patch(fc=OKABE["vermillion"], label="Transcriptome"), Patch(fc=OKABE["blue"], label="Metabolome (cross-half)")], 56, 66.5, ncol=1, fontsize=5)
        a.set_xticks(x); a.set_xticklabels(labs, rotation=90, fontsize=5); a.set_ylabel(yl); a.set_ylim(*lim)
        if p in ("c", "d"):
            a.set_yticks([0, 0.02, 0.04, 0.06])
    SD.add("SFig2", g)
    LED.put("SB_min_tumour", g.SB.min(), "{:.2f}"); LED.put("SB_max_tumour", g.SB.max(), "{:.2f}"); LED.put("SB_median_tumour", g.SB.median(), "{:.2f}")
    LED.put("depth_gain_median", g.depth_gain.median(), "{:.3f}"); LED.put("maldi_cross_norm_median", g.maldi_cross_norm.median(), "{:.2f}")
    LED.put("tx_norm_median_tumour", g.tx_norm_reliable.median(), "{:.2f}")
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S3 misregistration
def supp3(P, T, LED, SD, out):
    SC = T.tsv("Step8/m11_shift_curves.tsv"); RS = T.tsv("Step8/m11_registration_summary.tsv")
    F = K.Fig("SupplementaryFigure3", 78, kind="supp")
    F.region("a", 0, 0, 92, 78); a = F.ax("a", 12, 5, 76, 52)
    g = SC[(SC.level == "full") & (SC.res_um == 200)]
    for met, col, lab in (("tx", OKABE["vermillion"], "Transcriptome → metabolite"), ("self", OKABE["blue"], "Same metabolite, other pixel half"), ("maldi", OKABE["green"], "Other metabolites")):
        m = g[g.metric == met].groupby("delta_um").retention.median()
        for s_, gg in g[g.metric == met].groupby("sample"):
            a.plot(gg.delta_um, gg.retention, c=col, lw=0.4, alpha=0.35)
        a.plot(m.index, m.values, c=col, lw=1.4, marker="o", ms=2.5, label=lab)
    a.set_xlabel("Imposed shift of metabolite image (µm)"); a.set_ylabel("Retained $R^2$ (relative to no shift)"); a.set_ylim(0, 1.2); a.axhline(1, c="#999999", lw=0.4)
    _legend(F, "a", [Line2D([], [], c=OKABE["vermillion"], lw=1.4, label="Transcriptome → metabolite"), Line2D([], [], c=OKABE["blue"], lw=1.4, label="Same metabolite, other half"),
                     Line2D([], [], c=OKABE["green"], lw=1.4, label="Other metabolites")], 3, 70, ncol=3, fontsize=5)
    SD.add("SFig3a", g)
    F.region("b", 96, 0, 84, 78); b = F.ax("b", 110, 5, 66, 52)
    r = RS.reset_index(drop=True); x = np.arange(len(r))
    b.bar(x - 0.2, r.best_gain, 0.38, color="#BBBBBB", label="Naive best of 169 shifts"); b.bar(x + 0.2, r.cv_gain, 0.38, color=OKABE["vermillion"], label="Cross-fitted shift")
    b.set_xticks(x); b.set_xticklabels([short(s_) for s_ in r["sample"]], rotation=90, fontsize=5); b.set_ylabel("Gain in $R^2_{tx}$"); b.axhline(0, c="k", lw=0.4)
    _legend(F, "b", [Patch(fc="#BBBBBB", label="Naive best of 169 shifts"), Patch(fc=OKABE["vermillion"], label="Cross-fitted shift")], 112, 6.0, ncol=1, fontsize=5)
    SD.add("SFig3b", r)
    LED.put("naive_gain_median", r.best_gain.median(), "{:.2f}"); LED.put("cv_gain_median", r.cv_gain.median(), "{:.2f}"); LED.put("cv_gain_max", r.cv_gain.max(), "{:.2f}")
    LED.put("cv_gain_n_pos", int((r.cv_gain > 0.02).sum()), "{:d}"); LED.put("cv_n", len(r), "{:d}")
    LED.put("ret_tx_200_median", r.ret_tx_200.median(), "{:.2f}"); LED.put("ret_self_100_median", r.ret_self_100.median(), "{:.2f}")
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S4 oracle calibration details
def supp4(P, T, LED, SD, out):
    S = pd.concat([T.tsv("Step23/m24_oracle_summary.tsv"), T.tsv("Step24/m24_oracle_summary.tsv")], ignore_index=True).drop_duplicates(["dataset", "sample", "res_um"], keep="last")
    F = K.Fig("SupplementaryFigure4", 78, kind="supp")
    s2 = S[S.res_um == 200]; cols = [DESIGN_COL.get(d, "#555555") for d in s2.design]
    F.region("a", 0, 0, 60, 60); a = F.ax("a", 11, 5, 45, 42)
    a.scatter(s2.attainable_ridge_median, s2.attainable_projection_median, c=cols, s=12, edgecolor="white", lw=0.3); a.plot([0.4, 1], [0.4, 1], c="#999999", lw=0.5)
    a.set_xlabel("Attainable fraction, ridge oracle"); a.set_ylabel("Attainable fraction,\nprojection oracle"); a.set_xlim(0.4, 1); a.set_ylim(0.4, 1)
    F.region("b", 62, 0, 58, 60); b = F.ax("b", 73, 5, 45, 42)
    b.scatter(s2.null_q95_ridge, s2.null_q95_projection, c=cols, s=12, edgecolor="white", lw=0.3); b.plot([0, 0.6], [0, 0.6], c="#999999", lw=0.5)
    b.set_xlabel("Null 95th pct. of residual r, ridge"); b.set_ylabel("Null 95th pct., projection"); b.set_xlim(0, 0.6); b.set_ylim(0, 0.6)
    F.region("c", 122, 0, 58, 60); c = F.ax("c", 133, 5, 44, 42)
    e = S[S.dataset == "External-COSIE"].sort_values(["sample", "res_um"])
    for k, (col, lab) in enumerate((("ratio_median", "Observed"), ("attainable_median", "Attainable"), ("calibrated_median", "Calibrated"))):
        for j, r in enumerate(e.itertuples()):
            c.bar(j + (k - 1) * 0.26, getattr(r, col), 0.24, color=[OKABE["sky"], "#BBBBBB", OKABE["green"]][k])
    c.set_xticks(range(len(e))); c.set_xticklabels([f"H{r.sample[-1]}\n{int(r.res_um)} µm\nn={int(r.n_bins)}" for r in e.itertuples()], fontsize=5)
    c.set_ylim(0, 1.05); c.set_ylabel("Fraction")
    F.region("key", 0, 62, 180, 16, label=False)
    _legend(F, "key", [Line2D([], [], marker="o", ls="", ms=3, c=DESIGN_COL[d], label=DESIGN_LAB[d]) for d in DESIGN_ORDER], 3, 64, ncol=3, fontsize=5)
    _legend(F, "key", [Patch(fc=OKABE["sky"], label="Observed $R^2_{tx}/SB$"), Patch(fc="#BBBBBB", label="Attainable"), Patch(fc=OKABE["green"], label="Calibrated")], 124, 64, ncol=1, fontsize=5)
    SD.add("SFig4", S)
    for r in e.itertuples():
        LED.put(f"ext_{r.sample}_{int(r.res_um)}_att", r.attainable_median, "{:.2f}"); LED.put(f"ext_{r.sample}_{int(r.res_um)}_cal", r.calibrated_median, "{:.2f}")
        LED.put(f"ext_{r.sample}_{int(r.res_um)}_bins", int(r.n_bins), "{:d}"); LED.put(f"ext_{r.sample}_{int(r.res_um)}_nullq95", r.r_resid_null_q95, "{:.2f}")
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S5 adjacent-section penalty and context length
def supp5(P, T, LED, SD, out):
    DPN = T.tsv("Step11/m15_design_penalty.tsv"); PP = T.tsv("Step11/m15_pair_penalty.tsv"); CX = T.tsv("Step11/m14_context_summary.tsv"); V = T.json("Step11/m14_step11_verdict.json")
    F = K.Fig("SupplementaryFigure5", 132, kind="supp")
    F.region("a", 0, 0, 60, 62); a = F.ax("a", 12, 5, 44, 40)
    a.scatter(PP.vv_pc_corr, PP.ratio_cross_same_adjusted, s=14, c=OKABE["sky"], edgecolor="k", lw=0.3)
    a.set_xlabel("Transcriptomic similarity of the\npaired sections (PC correlation)"); a.set_ylabel("Adjacent / same-section\n$R^2_{tx}$/ceiling (shift-adjusted)"); a.set_ylim(0.4, 1.05)
    SD.add("SFig5a", PP)
    F.region("b", 62, 0, 56, 62); b = F.ax("b", 74, 5, 42, 40)
    x = np.arange(len(DPN))
    b.bar(x - 0.2, DPN.median_R2_same, 0.38, color=OKABE["blue"], label="Same section"); b.bar(x + 0.2, DPN.median_R2_cross, 0.38, color=OKABE["sky"], label="Adjacent pairing")
    for k, r in enumerate(DPN.itertuples()):
        b.text(k, max(r.median_R2_same, r.median_R2_cross) + 0.01, f"{100 * r.frac_features_same_gt_cross:.0f}%", ha="center", fontsize=5)
    b.set_xticks(x); b.set_xticklabels([s_.replace("v11l12-038-", "MALDI ") for s_ in DPN.msi_section], fontsize=5); b.set_ylabel("Median feature $R^2_{tx}$"); b.set_ylim(0, 0.32)
    _legend(F, "b", [Patch(fc=OKABE["blue"], label="Same section"), Patch(fc=OKABE["sky"], label="Adjacent pairing")], 64, 56.5, ncol=2, fontsize=5)
    SD.add("SFig5b", DPN)
    LED.put("feat_same_gt_cross_min", 100 * DPN.frac_features_same_gt_cross.min(), "{:.0f}"); LED.put("feat_same_gt_cross_max", 100 * DPN.frac_features_same_gt_cross.max(), "{:.0f}")
    LED.put("feat_same_gt_cross_mean", 100 * DPN.frac_features_same_gt_cross.mean(), "{:.0f}")
    F.region("c", 120, 0, 60, 62); c = F.ax("c", 132, 5, 44, 40)
    g = CX[(CX.level == "full") & (CX.res_um == 200)].copy(); g["grp"] = np.where(g.dataset == "SMA-cross", "Adjacent pairing\n(same mouse)", np.where(g.same_section, "Same section", "Adjacent / registered\n(human)"))
    grp = ["Same section", "Adjacent pairing\n(same mouse)", "Adjacent / registered\n(human)"]
    for k, gn in enumerate(grp):
        e = g[g.grp == gn]
        c.scatter(np.full(len(e), k) + np.linspace(-0.12, 0.12, len(e)), e.sigma_star_metab, s=10, c=OKABE["vermillion"], zorder=3)
        c.scatter(np.full(len(e), k) + 0.3, e.sigma_star_gene_jit, s=8, c="#999999", marker="s", zorder=3)
    c.set_xticks(range(len(grp))); c.set_xticklabels(grp, fontsize=5); c.set_yscale("log", base=2); c.set_yticks([200, 400, 800, 1600]); c.set_yticklabels(["200", "400", "800", "1600"])
    c.set_ylabel("Optimal smoothing σ* (µm)")
    _legend(F, "c", [Line2D([], [], marker="o", ls="", ms=3, c=OKABE["vermillion"], label="Metabolites"), Line2D([], [], marker="s", ls="", ms=3, c="#999999", label="Held-out genes (+ jitter)")],
            122, 58.5, ncol=2, fontsize=5)
    SD.add("SFig5c", g)
    F.region("d", 0, 66, 90, 66); d = F.ax("d", 12, 70, 74, 40)
    gg = g[g.n_lipid > 0].reset_index(drop=True); x = np.arange(len(gg))
    d.bar(x - 0.2, gg.sigma_star_small, 0.38, color=OKABE["orange"], label="Small molecules (m/z < 400)"); d.bar(x + 0.2, gg.sigma_star_lipid, 0.38, color=OKABE["blue"], label="Lipids")
    d.set_xticks(x); d.set_xticklabels([short(s_) for s_ in gg["sample"]], rotation=90, fontsize=5); d.set_ylabel("σ* (µm)")
    _legend(F, "d", [Patch(fc=OKABE["orange"], label="Small molecules (m/z < 400)"), Patch(fc=OKABE["blue"], label="Lipids")], 14, 69.0, ncol=2, fontsize=5)
    SD.add("SFig5d", gg)
    F.region("e", 94, 66, 86, 66); e = F.ax("e", 108, 70, 68, 40)
    e.bar(range(len(g)), g.gain_metab, 0.4, color=OKABE["vermillion"], label="Metabolites"); e.bar(np.arange(len(g)) + 0.4, g.gain_gene, 0.4, color="#999999", label="Held-out genes")
    e.set_xticks(np.arange(len(g)) + 0.2); e.set_xticklabels([short(s_) for s_ in g["sample"]], rotation=90, fontsize=5); e.set_ylabel("$R^2$ gain from transcriptomic context")
    _legend(F, "e", [Patch(fc=OKABE["vermillion"], label="Metabolites"), Patch(fc="#999999", label="Held-out genes")], 110, 69.0, ncol=2, fontsize=5)
    SD.add("SFig5e", g)
    v = V["full_all"]; LED.put("ctx_sigma_metab", v["median_sigma_metab"], "{:.0f}"); LED.put("ctx_sigma_gene", v["median_sigma_gene_jit"], "{:.0f}")
    LED.put("ctx_gain_metab", v["median_gain_metab"], "{:.2f}"); LED.put("ctx_gain_gene", v["median_gain_gene_jit"], "{:.3f}"); LED.put("ctx_small_lipid_p", v["stouffer_small_gt_lipid"], "{:.4f}")
    LED.put("ctx_small_lipid_p_adj", v["stouffer_adj"], "{:.4f}"); LED.put("ctx_frac_dir", 100 * v["frac_dir"], "{:.0f}")
    vs = V["full_same-section"]; LED.put("ctx_sigma_same", vs["median_sigma_metab"], "{:.0f}")
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S6 histology and variance budget
def supp6(P, T, LED, SD, out):
    H = T.tsv("Step12/m16_histology_summary.tsv"); GC = T.tsv("Step12/m16_group_contrib.tsv")
    g = H[(H.level == "full") & (H.res_um == 200) & (H.dataset != "SMA-cross")].reset_index(drop=True)
    F = K.Fig("SupplementaryFigure6", 130, kind="supp")
    x = np.arange(len(g)); labs = [short(s_) for s_ in g["sample"]]
    F.region("a", 0, 0, 90, 64); a = F.ax("a", 12, 5, 74, 38)
    a.bar(x - 0.2, g.delta_H, 0.38, color=OKABE["purple"], label="Metabolites"); a.bar(x + 0.2, g.delta_H_genes, 0.38, color="#999999", label="Held-out genes")
    a.scatter(x - 0.2, g.delta_H_null95, marker="_", c="k", s=25, lw=0.8, zorder=3, label="Rotation null, 95th percentile")
    a.set_xticks(x); a.set_xticklabels(labs, rotation=90, fontsize=5); a.set_ylabel("$R^2$ gain from H&E features"); a.axhline(0, c="k", lw=0.4)
    _legend(F, "a", [Patch(fc=OKABE["purple"], label="Metabolites"), Patch(fc="#999999", label="Held-out genes"), Line2D([], [], marker="_", ls="", c="k", ms=5, label="Rotation null, 95th percentile")],
            14, 4.0, ncol=3, fontsize=5)
    SD.add("SFig6a", g)
    F.region("b", 94, 0, 86, 64); b = F.ax("b", 106, 5, 70, 38)
    bot = np.zeros(len(g))
    for col, fc, lab in (("budget_local", OKABE["vermillion"], "Local transcriptome"), ("budget_context", OKABE["orange"], "Transcriptomic context"), ("budget_histo", OKABE["purple"], "Histology"),
                         ("budget_unexplained", "#D9D9D9", "Unexplained")):
        b.bar(x, g[col].clip(lower=0), 0.7, bottom=bot, color=fc, label=lab); bot += g[col].clip(lower=0).values
    b.set_xticks(x); b.set_xticklabels(labs, rotation=90, fontsize=5); b.set_ylabel("Share of reliable metabolite variance"); b.set_ylim(0, 1.0)
    _legend(F, "b", [Patch(fc=c_, label=l_) for c_, l_ in ((OKABE["vermillion"], "Local transcriptome"), (OKABE["orange"], "Context"), (OKABE["purple"], "Histology"), ("#D9D9D9", "Unexplained"))],
            106, 1.2, ncol=4, fontsize=5, columnspacing=0.6)
    SD.add("SFig6b", g[["dataset", "sample", "design", "budget_local", "budget_context", "budget_histo", "budget_unexplained"]])
    for col, key in (("budget_local", "local"), ("budget_context", "context"), ("budget_histo", "histo"), ("budget_unexplained", "unexpl")):
        LED.put(f"budget_{key}", 100 * g[col].mean(), "{:.0f}")
    LED.put("deltaH_median", g.delta_H.median(), "{:.3f}"); LED.put("deltaH_frac_null", 100 * (g.delta_H > g.delta_H_null95).mean(), "{:.0f}")
    F.region("c", 0, 68, 180, 62); c = F.ax("c", 30, 72, 110, 30)
    gc = GC[GC.level == "full"].pivot_table(index="group", columns="sample", values="median_drop_in_R2").reindex(columns=g["sample"])
    im = c.imshow(gc.values, cmap="Purples", aspect="auto", vmin=0, vmax=np.nanpercentile(gc.values, 98))
    c.set_yticks(range(len(gc))); c.set_yticklabels(gc.index, fontsize=5); c.set_xticks(range(gc.shape[1])); c.set_xticklabels([short(s_) for s_ in gc.columns], rotation=90, fontsize=5)
    c.tick_params(length=0, pad=1)
    _cbar(F, "c", 146, 80, 28, 1.3, im, "Drop in $R^2$ when group removed")
    SD.add("SFig6c", gc.reset_index())
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S7 neurotransmitter maps and robustness
def supp7(P, T, LED, SD, out):
    MS = T.tsv("Step13/m18_molecule_section.tsv"); RB = T.tsv("Step14/m18_robustness.tsv"); ST = T.tsv("Step17/m20_strict_targeted.tsv")
    F = K.Fig("SupplementaryFigure7", 158, kind="supp")
    F.region("a", 0, 0, 100, 86); a = F.ax("a", 30, 10, 50, 56)
    mols = [m for m in MS.molecule.unique()]
    piv = MS.pivot_table(index="molecule", columns="sample", values="rho_synthesis").reindex(mols)
    order_s = [s_ for s_ in MS["sample"].unique() if "085" in s_] + [s_ for s_ in MS["sample"].unique() if "109" in s_]
    piv = piv[order_s]; piv = piv.loc[piv.notna().any(axis=1)]
    im = a.imshow(piv.values, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
    a.set_yticks(range(len(piv))); a.set_yticklabels(piv.index, fontsize=5); a.set_xticks(range(piv.shape[1])); a.set_xticklabels([short(s_) for s_ in piv.columns], rotation=90, fontsize=5)
    a.tick_params(length=0, pad=1)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                a.text(j, i, K.num(v), ha="center", va="center", fontsize=5, color="white" if abs(v) > 0.3 else "black")
    cb = _cbar(F, "a", 83, 30, 15, 1.3, im, "r with own\nsynthesis genes"); cb.set_ticks([-0.4, 0, 0.4])
    SD.add("SFig7a", piv.reset_index())
    F.region("b", 104, 0, 76, 80); b = F.ax("b", 116, 10, 60, 50)
    g = ST[ST.molecule == "GABA"]
    for r in g.itertuples():
        x = 0 if r.brain_area == SN else 1
        b.scatter(x + np.random.default_rng(len(r.sample)).uniform(-0.08, 0.08), r.r, s=14, c=REGION_COL[r.brain_area], edgecolor="k" if r.p_dual_matched < 0.05 else "none", lw=0.6)
    b.set_xticks([0, 1]); b.set_xticklabels(["Substantia nigra", "Striatum"], fontsize=5.5); b.set_xlim(-0.5, 1.5); b.axhline(0, c="k", lw=0.4)
    b.set_ylabel("r, GABA vs $\\it{Gad1, Gad2, Slc32a1}$"); b.set_ylim(-0.1, 0.5)
    SD.add("SFig7b", g)
    F.region("c", 0, 90, 180, 68); c = F.ax("c", 14, 98, 162, 34)
    rb = RB.copy(); vs = list(dict.fromkeys(rb.variant)); xs = {v: i for i, v in enumerate(vs)}
    for s_, gg in rb.groupby("sample"):
        area = gg.brain_area.iloc[0]
        c.plot([xs[v] for v in gg.variant], gg.rho, c=REGION_COL[area], lw=0.5, alpha=0.6, marker="o", ms=2)
    c.set_xticks(range(len(vs))); c.set_xticklabels([variant_name(v) for v in vs], rotation=60, ha="right", rotation_mode="anchor", fontsize=5); c.set_ylabel("r, dopamine vs synthesis score\n(each section)")
    c.axhline(0, c="k", lw=0.4)
    _legend(F, "c", [Line2D([], [], c=REGION_COL[SN], marker="o", ms=2, lw=0.5, label="Substantia nigra sections"), Line2D([], [], c=REGION_COL[STR], marker="o", ms=2, lw=0.5, label="Striatum sections")],
            16, 92.0, ncol=2, fontsize=5)
    SD.add("SFig7c", rb)
    gs = ST[ST.molecule == "GABA"]; LED.put("gaba_str_sig", int(((gs.brain_area == STR) & (gs.p_dual_matched < 0.05)).sum()), "{:d}")
    LED.put("gaba_sn_sig", int(((gs.brain_area == SN) & (gs.p_dual_matched < 0.05)).sum()), "{:d}")
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S8 perturbation details and control molecules
def supp8(P, T, LED, SD, out):
    R = T.tsv("Step20/m22r_sections.tsv")
    F = K.Fig("SupplementaryFigure8", 132, kind="supp")
    g = R[(R["shift"].astype(str) == "True") & (R.res_um == 200)].reset_index(drop=True)
    F.region("a", 0, 0, 180, 60); a = F.ax("a", 14, 6, 162, 34)
    x = np.arange(len(g)); colm = {"dopamine": OKABE["black"], "serotonin": OKABE["sky"], "N-acetylaspartate": OKABE["purple"]}
    for k, r in enumerate(g.itertuples()):
        if np.isfinite(r.ci_lo):
            a.plot([k, k], [r.ci_lo, r.ci_hi], c=colm.get(r.molecule, "#555555"), lw=0.9)
        a.scatter(k, r.frac_explained, s=12, c=colm.get(r.molecule, "#555555"), zorder=3)
    a.set_xticks(x); a.set_xticklabels([f"{short(r.sample)}\n{r.molecule.replace('N-acetylaspartate', 'NAA')}" for r in g.itertuples()], rotation=90, fontsize=5)
    a.axhline(0, c="k", lw=0.4); a.set_ylabel("Explained fraction (95% CI)"); a.set_ylim(-1.0, 0.85)
    _legend(F, "a", [Line2D([], [], marker="o", ls="", ms=3, c=v, label=k) for k, v in colm.items()], 16, 2.0, ncol=3, fontsize=5)
    SD.add("SFig8a", g)
    F.region("b", 0, 64, 90, 68); b = F.ax("b", 14, 68, 72, 40)
    mk = {"dopamine": "o", "serotonin": "s", "N-acetylaspartate": "^"}
    for r in g.itertuples():
        b.scatter(r.mean_dSyn, r.mean_dY, marker=mk.get(r.molecule, "o"), c=REGION_COL[r.brain_area], s=14, edgecolor="white", lw=0.3)
    _legend(F, "b", [Line2D([], [], marker=v, ls="", ms=3, c="#555555", label=k.replace("N-acetylaspartate", "NAA")) for k, v in mk.items()], 16, 122.5, ncol=3, fontsize=5)
    _legend(F, "b",
            [Line2D([], [], marker="o", ls="", ms=3, c=REGION_COL[SN], label="Substantia nigra"), Line2D([], [], marker="o", ls="", ms=3, c=REGION_COL[STR], label="Striatum")], 16, 127.0, ncol=2, fontsize=5)
    b.axhline(0, c="k", lw=0.4); b.axvline(0, c="k", lw=0.4); b.set_xlabel("Mean Δ synthesis-gene score (intact − lesioned)"); b.set_ylabel("Mean Δ metabolite (s.d.)")
    SD.add("SFig8b", g)
    F.region("c", 94, 64, 86, 68); c = F.ax("c", 108, 68, 68, 34)
    ps = g[g.molecule == "dopamine"].reset_index(drop=True)
    c.bar(np.arange(len(ps)), ps.n_pairs, 0.6, color=[REGION_COL[a_] for a_ in ps.brain_area])
    c.set_xticks(range(len(ps))); c.set_xticklabels([short(s_) for s_ in ps["sample"]], rotation=90, fontsize=5); c.set_ylabel("Mirror-paired 200-µm bins\n(intact ↔ lesioned)")
    SD.add("SFig8c", ps)
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S9 GEM specificity and specification-curve details
def supp9(P, T, LED, SD, out):
    GS = T.tsv("Step7/m9_gem_summary.tsv"); SC = T.tsv("Step17/m20_spec_curve.tsv"); CA = T.tsv("Step17/m20_cptac_autonomy.tsv")
    F = K.Fig("SupplementaryFigure9", 128, kind="supp")
    F.region("a", 0, 0, 90, 62); a = F.ax("a", 12, 5, 74, 34)
    g = GS[GS.res_um == 200].copy(); lv = ["full", "detrended", "beyond_comp"]; smp = list(dict.fromkeys(g["sample"]))
    for k, l_ in enumerate(lv):
        e = g[g.level == l_].set_index("sample").reindex(smp)
        a.bar(np.arange(len(smp)) + (k - 1) * 0.26, 100 * e.frac_p05_gem, 0.24, color=[OKABE["vermillion"], OKABE["orange"], OKABE["yellow"]][k], label=l_)
    a.axhline(5, c="k", ls="--", lw=0.5); a.set_xticks(range(len(smp))); a.set_xticklabels([short(s_) for s_ in smp], rotation=90, fontsize=5); a.set_ylabel("% metabolites, own pathway P < 0.05")
    _legend(F, "a", [Patch(fc=c_, label=l_) for c_, l_ in zip([OKABE["vermillion"], OKABE["orange"], OKABE["yellow"]], ["Full model", "Detrended", "Beyond composition"])], 14, 4.0, ncol=3, fontsize=5)
    SD.add("SFig9a", g)
    F.region("b", 94, 0, 86, 62); b = F.ax("b", 108, 5, 68, 34)
    ca = CA[CA.level == "full"]
    b.hist(ca.pctl, bins=np.linspace(0, 1, 11), color=OKABE["grey"], edgecolor="white"); b.axhline(len(ca) / 10, c="k", ls="--", lw=0.5)
    b.set_xlabel("Own-pathway percentile, CPTAC GBM (n = 75)"); b.set_ylabel("Metabolites")
    SD.add("SFig9b", ca)
    F.region("c", 0, 66, 180, 62); c = F.ax("c", 40, 70, 90, 40)
    sc = SC[SC.level == "full"].pivot_table(index="subset", columns="predictor", values="rho", aggfunc="median")
    im = c.imshow(sc.values, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    c.set_yticks(range(len(sc))); c.set_yticklabels(sc.index.str.replace("_", " "), fontsize=5); c.set_xticks(range(sc.shape[1])); c.set_xticklabels(sc.columns.str.replace("_", " "), rotation=30, ha="right",
                                                                                                                                        rotation_mode="anchor", fontsize=5)
    c.tick_params(length=0, pad=1)
    for i in range(sc.shape[0]):
        for j in range(sc.shape[1]):
            c.text(j, i, K.num(sc.values[i, j]), ha="center", va="center", fontsize=5)
    _cbar(F, "c", 138, 80, 30, 1.3, im, "Median ρ (export share vs\nown-pathway specificity)")
    SD.add("SFig9c", sc.reset_index())
    return F.save(out["supp"], out["panels"])


# ------------------------------------------------------------------------------------------- S10 power
def supp10(P, T, LED, SD, out):
    PW = T.tsv("Step18/m21_power_grid.tsv")
    F = K.Fig("SupplementaryFigure10", 84, kind="supp")
    hyp = [("H1_NAA_tracks_neurons", "H1: NAA tracks neuron content"), ("H2_hypoxia_lowers_NAA_per_neuron", "H2: hypoxia lowers NAA per neuron"), ("H3_interaction", "H3: neuron × hypoxia interaction")]
    for k, (h, t_) in enumerate(hyp):
        p = "abc"[k]; F.region(p, k * 60, 0, 60, 70); a = F.ax(p, k * 60 + 11, 10, 46, 46)
        rec = T.json("Step18/m21_recommendation.json")[h]; d0 = rec["delta"]
        g = PW[(PW.hypothesis == h) & (np.isclose(PW.tau, 0.10)) & (np.isclose(PW.delta, d0))]
        g = g[np.isclose(g.alpha, g.alpha.min())]
        for res, col in ((100, OKABE["blue"]), (200, OKABE["vermillion"]), (400, OKABE["orange"])):
            for des, ls in (("same_section", "-"), ("adjacent_section", "--")):
                e = g[(g.res_um == res) & (g.design == des) & (g.regions_per_patient == 2)].sort_values("n_patients")
                a.plot(e.n_patients, e.power, c=col, ls=ls, lw=0.8)
        a.axhline(0.8, c="#999999", lw=0.5, ls=":"); a.set_ylim(0, 1.02); a.set_xlabel("Patients (2 regions each)"); a.set_ylabel("Power" if k == 0 else "")
        F.text(p, k * 60 + 11, 5.5, f"{t_}\n(δ = {d0:g} s.d.)", fontsize=5.5, va="center")
    F.region("key", 0, 72, 180, 12, label=False)
    _legend(F, "key", [Line2D([], [], c=OKABE["blue"], label="100 µm bins"), Line2D([], [], c=OKABE["vermillion"], label="200 µm bins"), Line2D([], [], c=OKABE["orange"], label="400 µm bins"),
                       Line2D([], [], c="k", ls="-", label="Same section"), Line2D([], [], c="k", ls="--", label="Adjacent sections")], 12, 75, ncol=5, fontsize=5)
    SD.add("SFig10", PW[(PW.tau == 0.10) & (PW.alpha == 0.025) & (PW.regions_per_patient == 2)])
    return F.save(out["supp"], out["panels"])
