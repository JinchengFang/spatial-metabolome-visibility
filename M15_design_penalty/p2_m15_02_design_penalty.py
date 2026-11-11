#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step11"); plt.rcParams.update({"font.size": 7})
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
RS_s, RS_x = rd(T("Step10"), "m11_registration_summary.tsv"), rd(TAB, "m11_registration_summary.tsv")
OS_s, OS_x = rd(T("Step10"), "m12_orthogonal_summary.tsv"), rd(TAB, "m12_orthogonal_summary.tsv")
RF_s, RF_x = rd(T("Step10"), "m12_residual_features.tsv"), rd(TAB, "m12_residual_features.tsv")
PAIRS, M14 = rd(TAB, "m15_cross_pairs.tsv"), rd(TAB, "m14_context_summary.tsv")
RS_adj = rd(T("Step8"), "m11_registration_summary.tsv")
if RS_s is None or RS_x is None or PAIRS is None:
    raise SystemExit("M11-01 results of same-section (Step10) or cross-pair (Step11) data / m15_cross_pairs.tsv missing")
RS_s = RS_s[(RS_s.res_um == a.res) & RS_s.dataset.astype(str).str.startswith("SMA-mouse")].copy()
RS_x = RS_x[(RS_x.res_um == a.res) & (RS_x.dataset == "SMA-cross")].copy()
RS_s["msi_section"] = RS_s["sample"].str.rsplit("_", n=1).str[0]
RS_x["pair"] = RS_x["sample"].str.rsplit("_", n=1).str[0]; RS_x = RS_x.merge(PAIRS[["pair", "msi_section", "visium_cross", "vv_pc_corr"]], on="pair", how="left")
metrics = ["R2_tx_0", "ceiling_SB", "tx_over_ceiling_0", "tx_over_ceiling_adjusted", "cv_gain", "ret_tx_100", "ret_self_100"]
rows = []
for sec, g in RS_x.groupby("msi_section"):
    s = RS_s[RS_s.msi_section == sec]
    if not len(s):
        continue
    s = s.iloc[0]; r = dict(msi_section=sec, sample_same=s["sample"], n_cross=len(g), vv_pc_corr=g.vv_pc_corr.mean())
    for m in metrics:
        r[f"{m}_same"], r[f"{m}_cross"] = float(s[m]), float(g[m].mean())
    r["ratio_cross_same"] = r["tx_over_ceiling_0_cross"] / max(r["tx_over_ceiling_0_same"], 1e-9)
    r["ratio_cross_same_adjusted"] = r["tx_over_ceiling_adjusted_cross"] / max(r["tx_over_ceiling_adjusted_same"], 1e-9)
    r["penalty_adjusted"] = 1 - r["ratio_cross_same_adjusted"]
    if OS_s is not None and OS_x is not None:
        os_ = OS_s[OS_s["sample"] == s["sample"]]; ox = OS_x[OS_x["sample"].isin(g["sample"])]
        r["frac_orth_same"] = float(os_.frac_orthogonal_reliable.iloc[0]) if len(os_) else np.nan
        r["frac_orth_cross"] = float(ox.frac_orthogonal_reliable.mean()) if len(ox) else np.nan
    # per-feature pairs: R²_tx of the same MALDI feature, same section vs cross pairs (mean over the two adjacent Visium sections)
    if RF_s is not None and RF_x is not None:
        fs = RF_s[RF_s["sample"] == s["sample"]].set_index("feature")
        fx = RF_x[RF_x["sample"].isin(g["sample"])].groupby("feature")[["R2_tx", "reliability_SB"]].mean()
        j = fs[["R2_tx", "reliability_SB"]].join(fx, rsuffix="_cross", how="inner").dropna()
        j = j[j.reliability_SB > 0.3]
        if len(j) >= 10:
            d = j.R2_tx - j.R2_tx_cross
            r.update(n_features_paired=len(j), median_R2_same=float(j.R2_tx.median()), median_R2_cross=float(j.R2_tx_cross.median()),
                     frac_features_same_gt_cross=float((d > 0).mean()), wilcoxon_p=float(wilcoxon(d, alternative="greater").pvalue))
            jj = j.reset_index(); jj.insert(0, "msi_section", sec); jj.to_csv(os.path.join(TAB, f"m15_feature_pairs_{sec}.tsv"), sep="\t", index=False)
    if M14 is not None:
        m_ = M14[M14.level == "full"]
        ms, mx = m_[m_["sample"] == s["sample"]], m_[m_["sample"].isin(g["sample"])]
        if len(ms) and len(mx):
            r.update(sigma_star_metab_same=float(ms.mean_sigma_metab.iloc[0]), sigma_star_metab_cross=float(mx.mean_sigma_metab.mean()),
                     gain_metab_same=float(ms.gain_metab.iloc[0]), gain_metab_cross=float(mx.gain_metab.mean()))
    rows.append(r)
DP = pd.DataFrame(rows)
# per cross pair: does the penalty grow with transcriptomic dissimilarity of the two sections (lower Visium-Visium PC correlation)?
PR = RS_x[["pair", "msi_section", "visium_cross", "vv_pc_corr", "tx_over_ceiling_adjusted"]].merge(
    RS_s[["msi_section", "tx_over_ceiling_adjusted"]].rename(columns={"tx_over_ceiling_adjusted": "same_adjusted"}), on="msi_section")
PR["ratio_cross_same_adjusted"] = PR.tx_over_ceiling_adjusted / PR.same_adjusted.clip(lower=1e-9)
PR.to_csv(os.path.join(TAB, "m15_pair_penalty.tsv"), sep="\t", index=False)
if not len(DP):
    raise SystemExit("no MALDI section shared between same-section and cross-pair results")
DP.to_csv(os.path.join(TAB, "m15_design_penalty.tsv"), sep="\t", index=False)
v = dict(n_msi_sections=int(len(DP)))
v["D46_adjacent_penalty"] = bool(((DP.ratio_cross_same_adjusted <= 0.75) & (DP.get("wilcoxon_p", pd.Series(1.0, index=DP.index)) < 0.05)).all())
v["median_ratio_cross_same_adjusted"] = float(DP.ratio_cross_same_adjusted.median()); v["median_penalty_adjusted"] = float(DP.penalty_adjusted.median())
if "frac_orth_cross" in DP:
    v["D47_orthogonal_partly_design"] = bool((DP.frac_orth_cross > DP.frac_orth_same).all())
    v["frac_orth_same_median"], v["frac_orth_cross_median"] = float(DP.frac_orth_same.median()), float(DP.frac_orth_cross.median())
# extrapolation: same-section equivalent of the adjacent-section tumour cohorts (order of magnitude only)
if RS_adj is not None:
    ra = RS_adj[RS_adj.res_um == a.res]
    if len(ra) and v["median_ratio_cross_same_adjusted"] > 0:
        eq = (ra.tx_over_ceiling_adjusted / v["median_ratio_cross_same_adjusted"]).clip(upper=1)
        v["tumor_adjacent_tx_over_ceiling_median"] = float(ra.tx_over_ceiling_adjusted.median())
        v["tumor_same_section_equivalent_median"] = float(eq.median()); v["tumor_same_section_equivalent_max"] = float(eq.max())
if "sigma_star_metab_cross" in DP:      # context-length inflation as a feature of adjacent-section mismatch (calibration of M14)
    v["D48_context_inflation_by_design"] = bool((DP.sigma_star_metab_cross > DP.sigma_star_metab_same).all())
    v["sigma_star_same_median"], v["sigma_star_cross_median"] = float(DP.sigma_star_metab_same.median()), float(DP.sigma_star_metab_cross.median())
    v["gain_same_median"], v["gain_cross_median"] = float(DP.gain_metab_same.median()), float(DP.gain_metab_cross.median())
if len(PR) >= 4:
    from scipy.stats import spearmanr
    rho, p_ = spearmanr(PR.vv_pc_corr, PR.ratio_cross_same_adjusted); v["pair_ratio_vs_section_similarity_rho"], v["pair_ratio_vs_section_similarity_p"] = float(rho), float(p_)
C.save_json(v, os.path.join(TAB, "m15_verdict.json"))
C.log("per pair:\n" + PR.round(3).to_string(index=False))
C.log("adjacent-section penalty:\n" + DP.round(3).T.to_string() + "\ndecisions: " + json.dumps(v, ensure_ascii=False, default=str))
# figure
fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
axs[3].scatter(PR.vv_pc_corr, PR.ratio_cross_same_adjusted, c="#de2d26", s=18)
for r in PR.itertuples():
    axs[3].text(r.vv_pc_corr, r.ratio_cross_same_adjusted, " " + r.pair[-5:], fontsize=5)
axs[3].axhline(1, c="k", ls="--", lw=.6); axs[3].set_xlabel("section similarity (Visium↔Visium PC correlation)"); axs[3].set_ylabel("adjacent / same (R²_tx / ceiling)")
axs[3].set_title("penalty vs section dissimilarity", fontsize=7)
for k_, (m_s, m_x, lab) in enumerate((("tx_over_ceiling_adjusted_same", "tx_over_ceiling_adjusted_cross", "R²_tx / noise ceiling (shift-adjusted)"),
                                     ("frac_orth_same", "frac_orth_cross", "fraction with reliable, structured\ntranscriptome-orthogonal residual"),
                                     ("sigma_star_metab_same", "sigma_star_metab_cross", "metabolite context length σ* (µm)"))):
    if m_s not in DP:
        axs[k_].axis("off"); continue
    for r in DP.itertuples():
        axs[k_].plot([0, 1], [getattr(r, m_s), getattr(r, m_x)], "o-", c="#555", ms=4)
        axs[k_].text(1.05, getattr(r, m_x), str(r.msi_section)[-2:], fontsize=5, va="center")
    axs[k_].set_xticks([0, 1]); axs[k_].set_xticklabels(["same section", "adjacent section\n(same mouse)"]); axs[k_].set_ylabel(lab); axs[k_].set_xlim(-.3, 1.4)
axs[0].set_title(f"adjacent-section penalty {100 * v['median_penalty_adjusted']:.0f}% (median)", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_design_penalty.png"), dpi=180); plt.close(fig)
