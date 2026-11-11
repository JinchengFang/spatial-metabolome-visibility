#!/usr/bin/env python3

import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step8"); R0 = a.res
T7 = os.path.join(P["P2"], "05_tables", "Step7")
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
RS, TS, GC = (rd(TAB, f) for f in ("m11_registration_summary.tsv", "m11_transport_summary.tsv", "m11_gem_consistency.tsv"))
CS, GS, SS = rd(T7, "m9_ceiling_summary.tsv"), rd(T7, "m9_gem_summary.tsv"), rd(T7, "m7_shapley_summary.tsv")

verdict, ev = {}, []
if RS is not None:
    r = RS[RS.res_um == R0]
    verdict["D34"] = f"{int(r.D34_misregistration_not_limiting.sum())}/{len(r)}"
    ev.append(dict(claim="Misregistration / adjacent-section offset does not explain the transcriptome-metabolome gap", decision="D34", status=verdict["D34"],
                   key_numbers=f"median R²_tx/ceiling after shift correction {r.tx_over_ceiling_adjusted.median():.2f}; perfect coupling retained at delta = 100 µm {r.ret_self_100.median():.2f}"))
if TS is not None:
    t = TS[TS.res_um == R0]
    f = t[(t.level == "full") & (t.predictor == "HVG_PC")]
    if len(f):
        f = f.iloc[0]; verdict.update(D35=bool(f.D35_not_transportable) if pd.notna(f.D35_not_transportable) else None, D36=bool(f.get("D36_coupling_reproducible", False)))
        ev.append(dict(claim="Gene -> metabolite mappings do not transfer between patients", decision="D35", status=str(verdict["D35"]),
                       key_numbers=f"R²_within {f.R2_within:.3f} vs R²_LOPO {f.R2_LOPO:.3f} (r_LOPO {f.r_LOPO:.2f})"))
        ev.append(dict(claim="Transcriptomic coupling profiles of metabolites are reproducible between patients", decision="D36", status=str(verdict["D36"]),
                       key_numbers=f"matched r {f.profile_r_matched:.2f} vs unmatched 95% {f.profile_r_mismatch_p95:.2f}; reproducible fraction {f.frac_reproducible:.0%}; "
                                   f"rank correlation of predictability {f.predictability_rank_rho:.2f}"))
if GC is not None:
    g = GC[(GC.res_um == R0) & (GC.level == "full")]
    if len(g):
        g = g.iloc[0]
        ev.append(dict(claim="GEM own-pathway specificity does not replicate between patients", decision="D33b", status=str(bool(g.poisson_p >= 0.05)),
                       key_numbers=f"significant in >= 2 patients: observed {g.observed_sig_in_ge2} vs expected {g.expected_by_chance:.2f} (p = {g.poisson_p:.2f}); "
                                   f"between-patient rho of percentiles {g.pctl_pairwise_rho:.2f}"))
if CS is not None:
    c = CS[(CS.res_um == R0) & (CS.level == "full")]
    ev.append(dict(claim="Metabolite measurements are reliable; the transcriptome reaches only a small part of the noise ceiling", decision="D30",
                   status=f"{int(c.D30_gap_not_noise.sum())}/{len(c)}",
                   key_numbers=f"median ceiling SB {c.SB.median():.2f}; median R²_tx/ceiling (reliable features) {c.tx_norm_reliable.median():.2f}, "
                               f"range {c.tx_norm_reliable.min():.2f}-{c.tx_norm_reliable.max():.2f}"))
    ev.append(dict(claim="Sequencing depth is not the bottleneck", decision="D31", status=f"{int(c.D31_depth_not_limiting.sum())}/{len(c)}",
                   key_numbers=f"median gain extrapolated to infinite depth {c.depth_gain.median():.3f}"))
    ev.append(dict(claim="Internal metabolite predictability is not shared pixel noise", decision="D32", status=f"{int(c.D32_internal_exceeds_tx.sum())}/{len(c)}",
                   key_numbers=f"median MALDI cross-half R² {c.R2_maldi_cross.median():.2f}; same-half - cross-half {c.shared_pixel_noise.median():.3f}"))
if GS is not None:
    g = GS[(GS.res_um == R0)]
    for lv in ("full", "beyond_comp"):
        e = g[g.level == lv]
        if len(e):
            ev.append(dict(claim=f"GEM own pathways are no better than random metabolic genes ({lv})", decision="D33", status=f"{int(e.D33_no_specificity.sum())}/{len(e)}",
                           key_numbers=f"median percentile {e.pctl_med_gem.median():.2f}; median fraction p < 0.05 {e.frac_p05_gem.median():.2f}; MDE w = {e.MDE_w.median()}"))
if SS is not None:
    s = SS[(SS.res_um == R0)]
    for lv in ("full", "detrended"):
        e = s[s.level == lv]
        if len(e):
            ev.append(dict(claim=f"Cell-state (Hallmark) programmes carry little unique information ({lv})", decision="D29 (negative)",
                           status=f"{int((~e.D29_state_encoded.astype(bool)).sum())}/{len(e)}",
                           key_numbers=f"median unique delta R² of states {e.unique_state_adj.median():.3f}; technical block {e.tech.median():.3f}; composition {e.comp.median():.3f}"))
EV = pd.DataFrame(ev)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m11_evidence_matrix.tsv"), sep="\t", index=False)
C.save_json(verdict, os.path.join(TAB, "m11_step8_verdict.json"))
C.log(f"-> {TAB}/m11_step8_verdict.json, m11_evidence_matrix.tsv ({len(EV)} rows)")
