#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step12")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
HS, CS = rd(TAB, "m16_histology_summary.tsv"), rd(TAB, "m16_conservation.tsv")
AS, AC = rd(TAB, "m17_asymmetry_summary.tsv"), rd(TAB, "m17_asymmetry_consistency.tsv")
ST = rd(TAB, "m17_soma_terminal.tsv"); EV4 = rd(T("Step11"), "m14_evidence_matrix_v4.tsv")
fv54 = os.path.join(TAB, "m17_soma_terminal_verdict.json"); V54 = json.load(open(fv54)) if os.path.exists(fv54) else {}
ev, verdict = [], {}

if HS is not None:
    f = HS[(HS.level == "full") & (HS.design != "SMA-cross-control")]
    if len(f):
        verdict["D49"] = bool(f.D49_histology_increment.mean() >= 0.7); verdict["D50"] = bool(f.D50_histology_explains_orthogonal.fillna(False).mean() >= 0.7)
        ev.append(dict(claim="H&E explains metabolite variation beyond transcriptome + context, and this increment is metabolite-specific (about 0 for held-out genes)",
                       decision="D49", status=str(verdict["D49"]),
                       key_numbers=f"median dH {f.delta_H.median():+.3f} (genes {f.delta_H_genes.median():+.3f}); samples meeting the criterion {f.D49_histology_increment.mean():.0%}"))
        ev.append(dict(claim="H&E explains part of the transcriptome-orthogonal reliable metabolite variation", decision="D50", status=str(verdict["D50"]),
                       key_numbers=f"median fraction of orthogonal reliable variation explained by H&E {f.frac_orth_explained.median():.2f}; "
                                   f"samples meeting the criterion {f.D50_histology_explains_orthogonal.fillna(False).mean():.0%}"))
if CS is not None and len(CS):
    cc = CS[CS.dataset == "ccRCC"] if (CS.dataset == "ccRCC").any() else CS
    verdict["D51"] = bool(cc.rho.median() > 0 and (cc.p < 0.05).mean() >= 0.5)
    ev.append(dict(claim="Histology-associated metabolites are conserved across patients", decision="D51", status=str(verdict["D51"]),
                   key_numbers=f"{cc.dataset.iloc[0]} median pairwise Spearman {cc.rho.median():.2f}; p < 0.05 {(cc.p < 0.05).mean():.0%}"))
if AS is not None:
    verdict["D52"] = bool(AS.D52_reproducible_asymmetry.mean() >= 2 / 3); verdict["D53"] = bool(AS.D53_asymmetry_orthogonal.mean() >= 2 / 3)
    verdict["D53g"] = bool(AS.D53g_gene_positive_control.any())
    ev.append(dict(claim="Unilateral lesioning produces reproducible metabolic hemispheric asymmetry", decision="D52", status=str(verdict["D52"]),
                   key_numbers=f"replicated fraction {', '.join(f'{x:.0%}' for x in AS.frac_replicated)}; cross-section rho "
                               f"{', '.join(f'{x:.2f}' for x in AC.rho) if AC is not None and len(AC) else 'NA'}"))
    ev.append(dict(claim="Lesion-induced metabolic asymmetry is mainly transcriptome-orthogonal (molecules covered by DHB / 9-AA)", decision="D53", status=str(verdict["D53"]),
                   key_numbers=f"median delta_E/delta_Y {', '.join(f'{x:.2f}' for x in AS.median_frac_asym_orthogonal)}"))
    ev.append(dict(claim="Positive control: the transcriptome itself responds reproducibly to lesioning", decision="D53g", status=str(verdict["D53g"]),
                   key_numbers=f"replicated genes {', '.join(str(x) for x in AS.n_genes_replicated)}"))
if ST is not None:
    verdict["D54"] = V54.get("D54_soma_vs_terminal")
    ev.append(dict(claim="Same molecule (dopaminergic): predictable by the transcriptome in the soma region, not in the terminal-only region", decision="D54",
                   status=str(verdict["D54"]),
                   key_numbers=f"relative predictability SN {V54.get('Substantia Nigra', {}).get('rel_pred', np.nan):.2f} vs STR {V54.get('Striatum', {}).get('rel_pred', np.nan):.2f}"))
EV = pd.concat([EV4 if EV4 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m16_evidence_matrix_v5.tsv"), sep="\t", index=False)
C.save_json(verdict, os.path.join(TAB, "m16_step12_verdict.json"))
C.log(f"-> {TAB}/m16_evidence_matrix_v5.tsv ({len(EV)} rows), m16_step12_verdict.json")
