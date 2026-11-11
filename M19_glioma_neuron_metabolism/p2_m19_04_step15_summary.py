#!/usr/bin/env python3

import argparse, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step15")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
rj = lambda d, f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else {}
A, B, CS, V = rd(TAB, "m19c_mouse_naa.tsv"), rd(TAB, "m19c_glioma_interaction.tsv"), rd(TAB, "m19c_cptac_sensitivity.tsv"), rj(TAB, "m19c_verdict.json")
EV7 = rd(T("Step14"), "m19_evidence_matrix_v7.tsv")
ev = []

if A is not None:
    ev += [dict(claim="Same-section mouse: NAA specifically co-localizes with the neuronal synthesis side (NAT8L + neuronal genes), not with the oligodendroglial "
                      "degradation side (ASPA + myelin genes)", decision="D58a", status=str(V.get("D58a_NAA_neuronal")),
                key_numbers=f"r neuronal side {A.neuronal_r.min():.2f}-{A.neuronal_r.max():.2f}; percentile of NAA among all peaks "
                            f"{A.NAA_rank_pct_neuronal.min():.2f}-{A.NAA_rank_pct_neuronal.max():.2f}"),
           dict(claim="Same-section mouse: NAA is explained by its synthesis side (R² neuronal > oligodendroglial), an 'autonomous' molecule like nigral dopamine",
                decision="D58b", status=str(V.get("D58b_attribution_neuronal")),
                key_numbers=f"R² {A.neuronal_R2.median():+.3f} vs {A.oligodendroglial_R2.median():+.3f} (median)")]
if B is not None:
    ev += [dict(claim="IDH-mutant: in tumour tissue NAA still tracks local neuron number (premise)", decision="D58e", status=str(V.get("D58e_IDHmut_NAA_tracks_neurons")),
                key_numbers=json.dumps([{k: d[k] for k in ("res_um", "beta_neuron", "p_neuron_pos")} for d in V.get("IDHmut_detail", [])])[:200]),
           dict(claim="IDH-mutant: infiltration lowers the NAA contribution per neuron (interaction < 0)", decision="D58c", status=str(V.get("D58c_IDHmut_interaction")),
                key_numbers=json.dumps([{k: d[k] for k in ("res_um", "beta_interaction", "p_int_neg")} for d in V.get("IDHmut_detail", [])])[:200]),
           dict(claim="GBM Visium: interaction < 0 (>= 4/6 patients)", decision="D58d", status=str(V.get("D58d_GBM_pass")), key_numbers=str(V.get("D58d_GBM_interaction_neg")))]
if CS is not None:
    ev.append(dict(claim="CPTAC: NAA decrease with the malignant programme (controlling for neuronal / oligodendroglial) is robust across 6 model variants and driven by "
                         "mesenchymal / hypoxia programmes", decision="D57b_robust", status=str(V.get("D57b_robust")),
                   key_numbers=f"{V.get('D57b_variants_pass')}; by programme: " + json.dumps(V.get("NAA_by_program"))[:160]))
EV = pd.concat([EV7 if EV7 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m19_evidence_matrix_v8.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m19_evidence_matrix_v8.tsv ({len(EV)} rows)")
