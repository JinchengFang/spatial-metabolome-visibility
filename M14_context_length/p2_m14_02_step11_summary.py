#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step11")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
SM, FR = rd(TAB, "m14_context_summary.tsv"), rd(TAB, "m14_context_features.tsv")
EV3 = rd(T("Step10"), "m13_evidence_matrix_v3.tsv")
if SM is None:
    raise SystemExit("m14_context_summary.tsv missing; run p2_m14_01_context_length.py first")
verdict, ev = {}, []

SM["design"] = np.select([SM.dataset == "SMA-mouse", SM.dataset == "SMA-cross"], ["same-section", "SMA-cross-control"], "adjacent-section")
SM = SM[SM.design != "SMA-cross-control"]       # cross pairings are the control of M15 and are not part of D43-D45
V15 = json.load(open(os.path.join(TAB, "m15_verdict.json"))) if os.path.exists(os.path.join(TAB, "m15_verdict.json")) else None
for lv in ("full", "detrended"):
    s = SM[SM.level == lv]
    for des in ("all", "same-section", "adjacent-section"):
        d = s if des == "all" else s[s.design == des]
        if not len(d):
            continue
        ok43 = (d.p_metab_gt_gene_jit < 0.05) & (d.p_metab_gt_gene_jit_adj < 0.05)
        ok45 = (d.gain_metab >= 0.02) & (d.gain_metab > d.gain_gene_jit)
        m = d.p_small_gt_lipid.notna() & d.p_small_gt_lipid_adj.notna()
        st = float(norm.sf(norm.isf(d[m].p_small_gt_lipid.clip(1e-12, 1 - 1e-12)).sum() / np.sqrt(m.sum()))) if m.sum() else np.nan
        sta = float(norm.sf(norm.isf(d[m].p_small_gt_lipid_adj.clip(1e-12, 1 - 1e-12)).sum() / np.sqrt(m.sum()))) if m.sum() else np.nan
        fr44 = float((d[m].mean_sigma_small > d[m].mean_sigma_lipid).mean()) if m.sum() else np.nan
        verdict[f"{lv}_{des}"] = dict(n=int(len(d)), D43=bool(ok43.mean() >= 0.7), frac_D43=float(ok43.mean()),
                                      D44=(bool(st < 0.05 and sta < 0.05 and fr44 >= 0.7) if m.sum() else None), stouffer_small_gt_lipid=st, stouffer_adj=sta, frac_dir=fr44,
                                      D45=bool(ok45.mean() >= 0.7), frac_D45=float(ok45.mean()),
                                      median_sigma_metab=float(d.sigma_star_metab.median()), median_sigma_gene_jit=float(d.sigma_star_gene_jit.median()),
                                      median_gain_metab=float(d.gain_metab.median()), median_gain_gene_jit=float(d.gain_gene_jit.median()))
CLAIMS = {"D43": "Metabolites integrate a wider transcriptomic context than registration error can explain",
          "D44": "Diffusible small molecules have a wider context than membrane lipids",
          "D45": "Spatially smoothing the transcriptome to sigma* significantly improves metabolite prediction"}
for dk, claim in CLAIMS.items():
    for des in ("all", "same-section"):
        v = verdict.get(f"full_{des}")
        if not v:
            continue
        num = {"D43": f"median metabolite sigma* {v['median_sigma_metab']:.0f} µm vs registration-matched genes {v['median_sigma_gene_jit']:.0f} µm; samples meeting the criterion {v['frac_D43']:.0%}",
               "D44": f"Stouffer p {v['stouffer_small_gt_lipid']:.3g} (smoothness-adjusted {v['stouffer_adj']:.3g}); same direction {v['frac_dir']:.0%}" if v["D44"] is not None
               else "too few features of both classes within samples",
               "D45": f"median gain {v['median_gain_metab']:+.3f} vs genes {v['median_gain_gene_jit']:+.3f}; samples meeting the criterion {v['frac_D45']:.0%}"}[dk]
        ev.append(dict(claim=claim + ("" if des == "all" else " (same section)"), decision=dk + ("" if des == "all" else "_same"), status=str(v[dk]), key_numbers=num))
if FR is not None:
    top = FR[(FR.kind == "metab") & (FR.level == "full")].sort_values("context_gain", ascending=False)
    top = top[["dataset", "sample", "feature", "mz", "annotation", "cls", "R2_local", "R2_best", "sigma_star", "context_gain", "smooth"]].head(30)
    top.to_csv(os.path.join(TAB, "m14_top_context_features.tsv"), sep="\t", index=False)
if V15 is not None:
    ev.append(dict(claim="The adjacent-section design itself significantly lowers transcriptome-explainable metabolite variation (identical tissue, animal and matrix)",
                   decision="D46", status=str(V15.get("D46_adjacent_penalty")),
                   key_numbers=f"median cross/same (shift-corrected R²_tx/ceiling) {V15.get('median_ratio_cross_same_adjusted', np.nan):.2f}; "
                               f"penalty {100 * V15.get('median_penalty_adjusted', np.nan):.0f}%"))
    if "D47_orthogonal_partly_design" in V15:
        ev.append(dict(claim="Part of the 'transcriptome-orthogonal variation' is created by the adjacent-section design", decision="D47",
                       status=str(V15["D47_orthogonal_partly_design"]),
                       key_numbers=f"orthogonal fraction same section {V15['frac_orth_same_median']:.2f} -> adjacent {V15['frac_orth_cross_median']:.2f}"))
    if "D48_context_inflation_by_design" in V15:
        ev.append(dict(claim="Adjacent-section mismatch inflates the metabolite 'context length' (calibration of M14)", decision="D48",
                       status=str(V15["D48_context_inflation_by_design"]),
                       key_numbers=f"sigma* same section {V15['sigma_star_same_median']:.0f} -> adjacent {V15['sigma_star_cross_median']:.0f} µm; "
                                   f"gain {V15['gain_same_median']:+.3f} -> {V15['gain_cross_median']:+.3f}"))
EV = pd.concat([EV3 if EV3 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m14_evidence_matrix_v4.tsv"), sep="\t", index=False)
C.save_json(verdict, os.path.join(TAB, "m14_step11_verdict.json"))
C.log(f"-> {TAB}/m14_evidence_matrix_v4.tsv ({len(EV)} rows), m14_step11_verdict.json\n" + pd.DataFrame(verdict).T.round(3).to_string())
