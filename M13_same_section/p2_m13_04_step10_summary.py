#!/usr/bin/env python3

import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step10"); R0 = a.res
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
PH = rd(TAB, "m13_physchem_summary.tsv")
RS_same, OS_same = rd(TAB, "m11_registration_summary.tsv"), rd(TAB, "m12_orthogonal_summary.tsv")
RS_adj, OS_adj, EV9 = rd(T("Step8"), "m11_registration_summary.tsv"), rd(T("Step9"), "m12_orthogonal_summary.tsv"), rd(T("Step9"), "m12_evidence_matrix_v2.tsv")
verdict, ev = {}, []

rows = []
for design, RS, OS in (("same-section", RS_same, OS_same), ("adjacent-section", RS_adj, OS_adj)):
    if RS is None:
        continue
    r = RS[RS.res_um == R0]
    for x in r.itertuples():
        o = OS[OS["sample"] == x.sample] if OS is not None else pd.DataFrame()
        rows.append(dict(design=design, dataset=x.dataset, sample=x.sample, ceiling_SB=x.ceiling_SB, R2_tx=x.R2_tx_0, tx_over_ceiling=x.tx_over_ceiling_0,
                         tx_over_ceiling_shift_adjusted=x.tx_over_ceiling_adjusted, cv_gain=x.cv_gain, ret_self_100=x.ret_self_100,
                         residual_SB=o.residual_SB.iloc[0] if len(o) else np.nan, frac_orthogonal_reliable=o.frac_orthogonal_reliable.iloc[0] if len(o) else np.nan))
DC = pd.DataFrame(rows)
if len(DC):
    DC.to_csv(os.path.join(TAB, "m13_design_comparison.tsv"), sep="\t", index=False)
    s_ = DC[DC.design == "same-section"]
    if len(s_):
        verdict["D41"] = bool(s_.tx_over_ceiling_shift_adjusted.median() < 0.5)
        verdict["D42"] = bool(s_.frac_orthogonal_reliable.median() >= 0.5) if s_.frac_orthogonal_reliable.notna().any() else None
        ev.append(dict(claim="In same-section data the transcriptome also reaches only a small part of the noise ceiling", decision="D41", status=str(verdict["D41"]),
                       key_numbers=f"same-section median R²_tx/ceiling {s_.tx_over_ceiling_shift_adjusted.median():.2f} "
                                   f"(adjacent sections {DC[DC.design == 'adjacent-section'].tx_over_ceiling_shift_adjusted.median():.2f})"))
        ev.append(dict(claim="In same-section data, transcriptome-orthogonal variation is also reliable and structured", decision="D42", status=str(verdict["D42"]),
                       key_numbers=f"median residual reliability {s_.residual_SB.median():.2f}; reliable and structured {s_.frac_orthogonal_reliable.median():.0%}"))
if PH is not None:
    c = PH[(PH.group == "ccRCC") & (PH.kind == "residual")].set_index("subset") if "group" in PH else PH[PH.kind == "residual"].set_index("subset")
    if len(c):
        verdict["D40"] = bool(c.D40_not_ionization_physchem.iloc[0])
        ev.append(dict(claim="Conserved transcriptome-orthogonal covariation is not caused by ionisation physicochemistry", decision="D40", status=str(verdict["D40"]),
                       key_numbers=f"cross-polarity rho {c.loc['cross_polarity', 'rho_median']:.2f}; partial Mantel rho {c.loc['partial_mantel', 'rho_median']:.2f}; "
                                   f"physicochemical R² {c.physchem_R2_of_mean_residual_cov.iloc[0]:.2f}" if "cross_polarity" in c.index else ""))
EV = pd.concat([EV9 if EV9 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m13_evidence_matrix_v3.tsv"), sep="\t", index=False)
C.save_json(verdict, os.path.join(TAB, "m13_step10_verdict.json"))
C.log(f"-> {TAB}/m13_design_comparison.tsv, m13_evidence_matrix_v3.tsv ({len(EV)} rows), m13_step10_verdict.json")
