#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step14")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
rj = lambda d, f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else {}
RB, RBV = rd(TAB, "m18_robustness_summary.tsv"), rj(TAB, "m18_robustness_verdict.json")
V19 = rj(TAB, "m19_verdict.json"); EV6 = rd(T("Step13"), "m18_evidence_matrix_v6.tsv")
ev = []

if RB is not None:
    ev.append(dict(claim="Soma vs terminal (D54 / D55) is robust to resolution, shift, detrending, single ions, single synthesis genes and target-cell definition",
                   decision="D54_D55_robust", status=str(RBV.get("robust_D54") and RBV.get("robust_D55")),
                   key_numbers=f"D54 {100 * RBV.get('frac_D54', np.nan):.0f}%, D55 {100 * RBV.get('frac_D55', np.nan):.0f}% of {RBV.get('n_variants')} variants"))
if V19:
    ev += [dict(claim="IDH-mutant section: after controlling for neuron number, NAA still decreases with local malignant fraction", decision="D57a",
                status=str(V19.get("D57a_IDHmut_NAA")), key_numbers=json.dumps(V19.get("D57a_detail"))[:180]),
           dict(claim="CPTAC (75 tumours): after controlling for neuronal / oligodendroglial programmes, NAA decreases with the malignant programme", decision="D57b",
                status=str(V19.get("D57b_CPTAC_NAA")), key_numbers=json.dumps(V19.get("D57b_detail"))[:180]),
           dict(claim="The per-neuron NAA decrease exceeds that of NAT8L transcription", decision="D57d", status=str(V19.get("D57d_NAA_beyond_NAT8L")),
                key_numbers=json.dumps(V19.get("D57d_detail"))[:180]),
           dict(claim="Positive control: GPC increases with malignant fraction (accumulates in OPC-like tumour cells)", decision="ctrl_GPC",
                status=str(V19.get("ctrl_GPC_positive")), key_numbers="")]
EV = pd.concat([EV6 if EV6 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m19_evidence_matrix_v7.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m19_evidence_matrix_v7.tsv ({len(EV)} rows)")
