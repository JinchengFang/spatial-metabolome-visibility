#!/usr/bin/env python3

import argparse, json, os, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step24"); ap.add_argument("--n-boot", type=int, default=20000)
ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(2024)
T = lambda *p: os.path.join(P["P2"], "05_tables", *p)
parts = [pd.read_csv(f, sep="\t") for f in (T("Step23", "m24_oracle_summary.tsv"), T(a.step, "m24_oracle_summary.tsv")) if os.path.exists(f)]
if not parts:
    raise SystemExit("Missing Step23/m24_oracle_summary.tsv: run run_step23.sh first")
S = pd.concat(parts, ignore_index=True).drop_duplicates(["dataset", "sample", "res_um"], keep="last"); S = S[S.res_um == a.res].copy()
FE = [pd.read_csv(f, sep="\t") for f in (T("Step23", "m24_oracle_features.tsv"), T(a.step, "m24_oracle_features.tsv")) if os.path.exists(f)]
FE = pd.concat(FE, ignore_index=True) if FE else pd.DataFrame()
# ---------------------------------------------------------------- subject / tissue
MOUSE = {"038": "mPD3", "109-a1": "mPD1", "109-b1": "mPD3", "109-c1": "mPD4", "085_a1": "mPD1", "085_b1": "mPD3", "085_c1": "mPD4"}


def subject(r):
    s = str(r["sample"])
    if r["dataset"].startswith("SMA"):
        for k, m in MOUSE.items():
            if k in s:
                return f"mouse {m}"
    if r["dataset"] == "External-COSIE":
        return f"COSIE {s}"          # the two sections come from separate files (brain2 / brain3) and are treated as two animals
    return f"{r['dataset']} {s}"


def tissue(r):
    s = str(r["sample"])
    if r["dataset"].startswith("SMA"):
        return "mouse substantia nigra" if "085" in s else "mouse striatum"
    return {"External-COSIE": "mouse hippocampus", "ccRCC": "human ccRCC", "GBM": "human glioblastoma", "IDH-mut": "human IDH-mutant glioma"}.get(r["dataset"], r["dataset"])


S["subject"] = S.apply(subject, axis=1); S["tissue"] = S.apply(tissue, axis=1)
S.to_csv(os.path.join(TAB, "m25_sections.tsv"), sep="\t", index=False)
GROUPS = {"same-section (mouse brain, Vicari)": S.design == "same-section", "external (mouse hippocampus, another lab)": S.design == "external (another lab)",
          "adjacent-section (human tumours)": S.design == "adjacent-section", "registered cells (IDH-mutant glioma)": S.design == "registered-cells",
          "all real designs": ~S.design.astype(str).str.contains("artificial")}


def cluster_boot(g, col):
    """Two-stage cluster bootstrap: subjects, then sections within subjects; statistic = mean of subject means."""
    subs = {k: v[col].values for k, v in g.groupby("subject")}; keys = list(subs); est = float(np.mean([np.mean(v) for v in subs.values()]))
    if len(keys) < 3:                                                          # no bootstrap CI with < 3 subjects (external data: 2 animals)
        return est, np.nan, np.nan, len(keys), len(g)
    bs = np.empty(a.n_boot)
    for b in range(a.n_boot):
        ks = rng.choice(len(keys), len(keys), replace=True)
        bs[b] = np.mean([np.mean(rng.choice(subs[keys[k]], len(subs[keys[k]]), replace=True)) for k in ks])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi), len(keys), len(g)


rows = []
for gname, m in GROUPS.items():
    g = S[m]
    if not len(g):
        continue
    for col in ("calibrated_median", "ratio_median", "attainable_median", "r_resid_median"):
        est, lo, hi, ns, nsec = cluster_boot(g, col)
        rows.append(dict(group=gname, metric=col, estimate=est, ci_lo=lo, ci_hi=hi, n_subjects=ns, n_sections=nsec))
H = pd.DataFrame(rows); H.to_csv(os.path.join(TAB, "m25_headline_table.tsv"), sep="\t", index=False)
C.log("Headline (two-stage cluster bootstrap 95% CI):\n" + H[H.metric == "calibrated_median"].round(3).to_string(index=False))
# ---------------------------------------------------------------- within-mouse design penalty
X = S[S.design.astype(str).str.contains("artificial")].copy(); pairs = []
for r in X.itertuples():
    msi, rest = str(r.sample).split("x", 1); mat = rest.split("_", 1)[1] if "_" in rest else ""
    same = S[(S.design == "same-section") & (S["sample"] == f"{msi}_{mat}")]
    if len(same):
        pairs.append(dict(maldi_section=msi, matrix=mat, visium_from=rest.split("_")[0], cal_same=float(same.calibrated_median.iloc[0]), cal_cross=float(r.calibrated_median),
                          ratio_same=float(same.ratio_median.iloc[0]), ratio_cross=float(r.ratio_median), rE_same=float(same.r_resid_median.iloc[0]), rE_cross=float(r.r_resid_median)))
DP = pd.DataFrame(pairs)
if len(DP):
    DP["diff"] = DP.cal_cross - DP.cal_same; DP["retained"] = DP.cal_cross / DP.cal_same
    DP.to_csv(os.path.join(TAB, "m25_design_pairs.tsv"), sep="\t", index=False)
    per_sec = DP.groupby("maldi_section")["diff"].mean()
    C.log("Within-mouse pairs (calibrated ratio, same section -> adjacent):\n" + DP[["maldi_section", "matrix", "visium_from", "cal_same", "cal_cross", "retained"]].round(3).to_string(index=False))
# ---------------------------------------------------------------- descriptive: striatum vs substantia nigra (same mice, FMP-10)
fm = S[(S.design == "same-section") & S["sample"].str.contains("FMP10")]
reg = fm.pivot_table(index="subject", columns="tissue", values="calibrated_median") if len(fm) else pd.DataFrame()
# ---------------------------------------------------------------- criteria and headline JSON
gc = lambda g: H[(H.group == g) & (H.metric == "calibrated_median")]
allr = gc("all real designs")
V = dict(primary_metric="calibrated ratio = (cross-validated R²_tx / split-half Spearman–Brown reliability) ÷ attainable fraction under perfect linear transcriptomic determination; 200 µm; reliable features (SB ≥ 0.7)")
for gname in GROUPS:
    g = gc(gname)
    if len(g):
        x = g.iloc[0]; V[gname] = dict(calibrated=round(x.estimate, 3), ci=[None if np.isnan(x.ci_lo) else round(x.ci_lo, 3), None if np.isnan(x.ci_hi) else round(x.ci_hi, 3)],
                                       unexplained=round(1 - x.estimate, 3), n_subjects=int(x.n_subjects), n_sections=int(x.n_sections))
V["D68b_headline_upper_CI_below_0.75"] = bool(len(allr) and allr.iloc[0].ci_hi < 0.75) if len(allr) else "not testable"
if len(DP):
    V["D68a_within_mouse_design_penalty"] = bool((DP["diff"] < 0).sum() >= 5 and (per_sec < 0).all() and len(per_sec) >= 3)
    V["design_pairs"] = dict(n_pairs=len(DP), n_lower=int((DP["diff"] < 0).sum()), median_retained=round(float(DP.retained.median()), 3),
                             per_maldi_section_diff={k: round(float(v), 3) for k, v in per_sec.items()})
else:
    V["D68a_within_mouse_design_penalty"] = "not testable (no SMA-cross calibration: run M24-01 --datasets SMAx --step Step24)"
if len(reg) and reg.shape[1] == 2:
    V["descriptive_striatum_vs_nigra_FMP10"] = {k: {c: round(float(v), 3) for c, v in row.items()} for k, row in reg.iterrows()}
V["n_sections_total"] = int((~S.design.astype(str).str.contains("artificial")).sum()); V["n_datasets"] = int(S[~S.design.astype(str).str.contains("artificial")].dataset.nunique())
C.save_json(V, os.path.join(TAB, "m25_headline.json"))
C.log("Criteria: " + json.dumps({k: V[k] for k in V if k.startswith("D68")}))
# ---------------------------------------------------------------- figure (unified metric)
order = ["same-section", "adjacent-section (same mouse, artificial)", "external (another lab)", "registered-cells", "adjacent-section"]
lab = {"same-section": "same section\n(mouse brain)", "adjacent-section (same mouse, artificial)": "adjacent pairing,\nsame mouse", "external (another lab)": "external lab\n(hippocampus)",
       "registered-cells": "registered cells\n(IDH-mut glioma)", "adjacent-section": "adjacent sections\n(human tumours)"}
col = {"same-section": "#08519c", "adjacent-section (same mouse, artificial)": "#9ecae1", "external (another lab)": "#31a354", "registered-cells": "#756bb1", "adjacent-section": "#e6550d"}
S2 = S.copy(); S2["o"] = S2.design.map({d: i for i, d in enumerate(order)}); S2 = S2.sort_values(["o", "tissue", "sample"]).reset_index(drop=True)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, max(4, 0.22 * len(S2) + 1.5)), gridspec_kw=dict(width_ratios=[3, 1.3]))
y = np.arange(len(S2))[::-1]
for yy, r in zip(y, S2.itertuples()):
    if len(FE):
        fv = FE[(FE["sample"] == r.sample) & (FE.res_um == a.res)].calibrated.dropna()
        if len(fv):
            q1, q3 = np.percentile(fv, [25, 75]); ax.plot([q1, q3], [yy, yy], c=col.get(r.design, "k"), lw=1, alpha=.6)
    ax.scatter(r.calibrated_median, yy, c=col.get(r.design, "k"), s=22, marker="o" if "artificial" not in r.design else "D", zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f"{r.tissue.replace('mouse ', '').replace('human ', '')} · {str(r.sample).replace('v11l12-', '').replace('v11t17-', '')}" for r in S2.itertuples()], fontsize=5.5)
for d in order:
    idx = np.where(S2.design == d)[0]
    if len(idx):
        ax.text(1.02, y[idx].mean(), lab[d], transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", color=col[d])
ax.axvline(1, c="#999", lw=.6); ax.axvline(0.75, c="k", ls="--", lw=.6); ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("calibrated fraction of reliable metabolite variance explained by the transcriptome\n(point: median over features; line: interquartile range)")
gg = H[(H.metric == "calibrated_median") & (H.group != "all real designs")].reset_index(drop=True); gg = pd.concat([gg, H[(H.metric == "calibrated_median") & (H.group == "all real designs")]], ignore_index=True)
for k, r in enumerate(gg.itertuples()):
    yy = len(gg) - k; ax2.errorbar(r.estimate, yy, xerr=None if np.isnan(r.ci_lo) else [[r.estimate - r.ci_lo], [r.ci_hi - r.estimate]], fmt="D", c="k" if r.group == "all real designs" else "#555", ms=5, capsize=2)
    ax2.text(1.03, yy, f"{r.group}\n{r.estimate:.2f}" + ("" if np.isnan(r.ci_lo) else f" [{r.ci_lo:.2f}, {r.ci_hi:.2f}]") + f"  ({r.n_subjects} subj / {r.n_sections} sec)",
             transform=ax2.get_yaxis_transform(), fontsize=5, va="center")
ax2.set_yticks([]); ax2.set_xlim(0, 1); ax2.axvline(0.75, c="k", ls="--", lw=.6); ax2.set_xlabel("group mean, 95% cluster-bootstrap CI\n(subjects → sections)"); ax2.set_ylim(0.3, len(gg) + 0.7)
fig.suptitle("The transcriptome explains a minority of reliable spatial metabolite variance in every design, after correcting for estimator attenuation", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "unified_calibrated.png"), dpi=200); plt.close(fig)
C.log(f"-> m25_headline.json, unified_calibrated.png ({time.time() - t0:.0f} s)")
