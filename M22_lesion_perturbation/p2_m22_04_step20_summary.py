#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step20")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
V = json.load(open(os.path.join(TAB, "m22r_verdict.json"))) if os.path.exists(os.path.join(TAB, "m22r_verdict.json")) else {}
evp = os.path.join(P["P2"], "05_tables", "Step19", "m22_evidence_matrix_v11.tsv"); EV11 = pd.read_csv(evp, sep="\t") if os.path.exists(evp) else pd.DataFrame()
sn, st = V.get("SN_animals", []), V.get("STR_animals", [])
fmt = lambda L_: "; ".join(f"{d['mouse']} {d['frac_explained']:.2f} [{d['ci_lo']:.2f}, {d['ci_hi']:.2f}] P={d['p_frac']:.3f}" for d in L_ if d.get("frac_explained") is not None)
ev = [dict(claim="Mirror-pair perturbation: in SN, synthesis genes explain a substantial part of the lesion-induced dopamine asymmetry (block-permutation null, Stouffer combination)",
           decision="D63a", status=str(V.get("D63a_SN_fraction_positive")), key_numbers=f"Stouffer P = {V.get('SN_stouffer_p', np.nan):.3g}; {fmt(sn)}"[:200]),
      dict(claim="In each animal, this fraction is lower in STR than in SN", decision="D63b", status=str(V.get("D63b_STR_below_SN_each_animal")), key_numbers=fmt(st)[:200]),
      dict(claim="The SN-STR contrast is robust across 6 shift x bin-size variants", decision="D63c", status=str(V.get("D63c_robust")),
           key_numbers=f"{sum(s['contrast_holds'] for s in V.get('sensitivity', []))}/{len(V.get('sensitivity', []))} variants hold")]
EV = pd.concat([EV11, pd.DataFrame(ev)], ignore_index=True); EV.to_csv(os.path.join(TAB, "m22_evidence_matrix_v12.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m22_evidence_matrix_v12.tsv ({len(EV)} rows)")
