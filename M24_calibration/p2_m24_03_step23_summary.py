#!/usr/bin/env python3

import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step23")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
rj = lambda p: json.load(open(p)) if os.path.exists(p) else {}
O, X = rj(os.path.join(TAB, "m24_oracle_verdict.json")), rj(os.path.join(TAB, "m24x_verdict.json"))
S = pd.read_csv(os.path.join(TAB, "m24_oracle_summary.tsv"), sep="\t") if os.path.exists(os.path.join(TAB, "m24_oracle_summary.tsv")) else pd.DataFrame()
evs = sorted(glob.glob(os.path.join(P["P2"], "05_tables", "Step*", "*evidence_matrix_v*.tsv")), key=lambda f: int(re.search(r"_v(\d+)\.tsv", f).group(1)))
evs = [f for f in evs if int(re.search(r"_v(\d+)\.tsv", f).group(1)) <= 14] or evs          # the matrix this step extends (v14)
EVp = pd.read_csv(evs[-1], sep="\t") if evs else pd.DataFrame(columns=["claim", "decision", "status", "key_numbers"]); vnum = int(re.search(r"_v(\d+)\.tsv", evs[-1]).group(1)) if evs else 14
NEW = ["D66a", "D66b", "D66c", "D67a", "D67b", "D67c"]; EVp = EVp[~EVp.decision.astype(str).isin(NEW)]
f2 = lambda x: "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
j2 = lambda s: ", ".join(f2(x) for x in s)
ext2 = S[(S.dataset == "External-COSIE") & (S.res_um == 200)] if len(S) else S; ext4 = S[(S.dataset == "External-COSIE") & (S.res_um == 400)] if len(S) else S
inn = S[(S.dataset != "External-COSIE")] if len(S) else S
bydes = lambda col: "; ".join(f"{d} {g[col].median():.2f} (n={len(g)})" for d, g in inn.groupby("design")) if len(inn) else ""
PS = X.get("per_section", {})
ev = [dict(claim="The gap is not finite-sample attenuation: after calibrating by the attainable fraction under perfectly transcriptome-determined simulations, "
                 "the transcriptome still reaches only part (calibrated fraction < 0.75)", decision="D66a", status=str(O.get("D66a_gap_not_attenuation")),
           key_numbers=(f"external 200 µm attainable {j2(ext2.attainable_median)}, calibrated {j2(ext2.calibrated_median)}; internal calibrated median {bydes('calibrated_median')}; "
                        f"internal fraction < 0.75 {f2(O.get('internal_frac_calibrated_lt_0.75'))}") if len(S) else ""),
      dict(claim="Reproducibility of orthogonal residuals is not caused by the two halves sharing one prediction (observed r_E > 95th percentile of perfect-determination null)",
           decision="D66b", status=str(O.get("D66b_residual_not_shared_prediction")),
           key_numbers=(f"external r_E {j2(ext2.r_resid_median)} vs null 95% {j2(ext2.r_resid_null_q95)}; internal fraction {f2(O.get('internal_frac_rE_gt_null95'))}") if len(S) else ""),
      dict(claim="The lower external 400-µm ratio is finite-sample attenuation (attainable < 0.6 and calibrated within 0.25 of 200 µm)", decision="D66c",
           status=str(O.get("D66c_400um_is_attenuation")),
           key_numbers=(f"400 µm bins {', '.join(str(int(x)) for x in ext4.n_bins)}; attainable {j2(ext4.attainable_median)}; calibrated {j2(ext4.calibrated_median)} "
                        f"(200 µm {j2(ext2.calibrated_median)})") if len(ext4) else ""),
      dict(claim="External data: stronger models (100 principal components / kernel ridge) do not close the gap", decision="D67a", status=str(X.get("D67a_capacity_does_not_close_gap")),
           key_numbers="; ".join(f"{k} 30 PCs {f2(v['capacity_ridge30'])} -> best pre-specified {f2(v['capacity_best'])} ({v['capacity_best_model']}); "
                                 f"best including post hoc kernel models {f2(v.get('capacity_best_all'))} ({v.get('capacity_best_all_model')})" for k, v in PS.items())
           + f"; post hoc criterion (all models < 0.75): {X.get('D67a_posthoc_all_models_below_0.75')}"),
      dict(claim="External data: even the best of 75 registration perturbations (±200 µm × ±3°) does not close the gap", decision="D67b",
           status=str(X.get("D67b_registration_does_not_close_gap")),
           key_numbers="; ".join(f"{k} original {f2(v['registration_original'])} -> best {f2(v['registration_best'])} {v['registration_best_at']}" for k, v in PS.items())),
      dict(claim="External data: after removing homologous peaks (within-bin pixel residual correlation), the internal metabolome reference still exceeds the transcriptome",
           decision="D67c", status=str(X.get("D67c_internal_reference_not_redundancy")),
           key_numbers="; ".join(f"{k} R²_tx {f2(v['R2_tx'])}; R²_maldi {f2(v['R2_maldi_all'])} -> pruned {f2(v['R2_maldi_pruned'])} -> strict {f2(v['R2_maldi_pruned_strict'])}"
                                 for k, v in PS.items()))]
EV = pd.concat([EVp, pd.DataFrame(ev)], ignore_index=True); evname = f"m24_evidence_matrix_v{vnum + 1}.tsv"; EV.to_csv(os.path.join(TAB, evname), sep="\t", index=False)
# ---------------------------------------------------------------- composite figure of the external analyses
panels = [os.path.join(P["P2"], "04_figures", "Step22", "m23x_registration.png"), os.path.join(P["P2"], "04_figures", "Step22", "m23x_framework.png"),
          os.path.join(FIG, "m24_oracle.png"), os.path.join(FIG, "m24x_robustness.png")]
panels = [p for p in panels if os.path.exists(p)]
if panels:
    fig, axs = plt.subplots(len(panels), 1, figsize=(10, 3.4 * len(panels)))
    for ax_, p, lab in zip(np.atleast_1d(axs), panels, "abcd"):
        ax_.imshow(plt.imread(p)); ax_.axis("off"); ax_.text(-0.01, 1.0, lab, transform=ax_.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "external_calibration_panels.png"), dpi=150); plt.close(fig)
C.log(f"-> {TAB}/{evname} ({len(EV)} rows)")
