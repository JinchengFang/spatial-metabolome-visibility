#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step19")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
V = json.load(open(os.path.join(TAB, "m22_verdict.json"))) if os.path.exists(os.path.join(TAB, "m22_verdict.json")) else {}
evp = os.path.join(P["P2"], "05_tables", "Step17", "m20_evidence_matrix_v10.tsv"); EV10 = pd.read_csv(evp, sep="\t") if os.path.exists(evp) else pd.DataFrame()
fe = V.get("frac_explained_median", {})
ev = [dict(claim="Lesion perturbation (mirror pairing, anatomically matched): in SN, dopamine loss scales with synthesis-gene loss", decision="D62a",
           status=str(V.get("D62a_SN_pair_coupling")),
           key_numbers=json.dumps([{k: d.get(k) for k in ("mouse", "r_dY_dSyn", "p_one_sided")} for d in V.get("dopamine_detail", [])
                                   if d.get("brain_area") == "Substantia Nigra"])[:180]),
      dict(claim="In STR, dopamine is largely lost but unrelated to synthesis-gene change (terminal region)", decision="D62b", status=str(V.get("D62b_STR_no_coupling")),
           key_numbers=f"median fraction of the asymmetry explained by synthesis genes SN {fe.get('SN', np.nan):.2f} vs STR {fe.get('STR', np.nan):.2f}"),
      dict(claim="Control: no paired coupling for 5-HT", decision="D62c", status=str(V.get("D62c_5HT_control")), key_numbers="")]
EV = pd.concat([EV10, pd.DataFrame(ev)], ignore_index=True); EV.to_csv(os.path.join(TAB, "m22_evidence_matrix_v11.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m22_evidence_matrix_v11.tsv ({len(EV)} rows)")
