#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step13")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
M = rd(TAB, "m18_molecule_section.tsv"); EV5 = rd(T("Step12"), "m16_evidence_matrix_v5.tsv"); CS = rd(T("Step12"), "m16_conservation.tsv")
fv = os.path.join(TAB, "m18_verdict.json"); V = json.load(open(fv)) if os.path.exists(fv) else {}
ev = []

if M is not None:
    da = V.get("dopamine", {}); sn, st = da.get("Substantia Nigra", {}), da.get("Striatum", {})
    ev += [dict(claim="Same molecule (dopamine): co-localizes with synthesis genes and is predictable in the SN, where its synthesis mRNA is, but not in the terminal-only STR",
                decision="D54", status=str(V.get("D54_soma_vs_terminal")),
                key_numbers=f"r with TH/DDC/SLC6A3/SLC18A2 SN {sn.get('rho', np.nan):.2f} (significant {sn.get('frac_syn_sig', np.nan):.0%}) vs STR {st.get('rho', np.nan):.2f} "
                            f"({st.get('frac_syn_sig', np.nan):.0%}); predictability percentile {sn.get('pct', np.nan):.2f} vs {st.get('pct', np.nan):.2f}"),
           dict(claim="The transcripts explaining dopamine switch with compartment: synthesis genes in SN (autonomous), target-cell (MSN) genes in STR (non-autonomous)",
                decision="D55", status=str(V.get("D55_attribution")), key_numbers=json.dumps(V.get("D55_detail", {}))[:200]),
           dict(claim="Negative control: 5-HT, without synthesizing somata in either region, shows no gene-specific correlation in either region", decision="ctrl_5HT",
                status=str(V.get("neg_control_5HT_no_gene_association")), key_numbers=""),
           dict(claim="Positive control: GABA co-localizes with GAD1/GAD2/SLC32A1 in both regions (GABAergic somata in both)", decision="D54c",
                status=str(V.get("D54c_GABA_control")),
                key_numbers=f"r SN {V.get('GABA', {}).get('Substantia Nigra', {}).get('rho', np.nan):.2f} / STR {V.get('GABA', {}).get('Striatum', {}).get('rho', np.nan):.2f}"),
           dict(claim="Lesion-induced dopamine asymmetry is more 'transcriptome-orthogonal' in STR than in SN", decision="D54d", status=str(V.get("D54d_lesion_orthogonal_str_gt_sn")),
                key_numbers=f"orthogonal fraction STR {st.get('frac_orth', np.nan):.2f} vs SN {sn.get('frac_orth', np.nan):.2f}")]
if CS is not None and len(CS):
    per = CS.groupby("dataset").agg(rho=("rho", "median"), frac_p05=("p", lambda x: float((x < 0.05).mean())), n=("rho", "size")).reset_index()
    ev.append(dict(claim="Histology-associated metabolites are conserved across samples (per dataset)", decision="D51_by_dataset",
                   status="; ".join(f"{r.dataset}:{r.rho > 0 and r.frac_p05 >= 0.5}" for r in per.itertuples()),
                   key_numbers="; ".join(f"{r.dataset} rho {r.rho:.2f}" for r in per.itertuples())))
EV = pd.concat([EV5 if EV5 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m18_evidence_matrix_v6.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m18_evidence_matrix_v6.tsv ({len(EV)} rows)\ndecisions: " + json.dumps({k: v for k, v in V.items() if not isinstance(v, dict)}, default=str))
