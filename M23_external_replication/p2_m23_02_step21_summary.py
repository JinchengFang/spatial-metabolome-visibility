#!/usr/bin/env python3

import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step21")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
T = lambda *p: os.path.join(P["P2"], "05_tables", *p)
V = json.load(open(os.path.join(TAB, "m23_verdict.json"))) if os.path.exists(os.path.join(TAB, "m23_verdict.json")) else {}
evs = sorted(glob.glob(T("Step*", "*evidence_matrix_v*.tsv")), key=lambda f: int(re.search(r"_v(\d+)\.tsv", f).group(1)))
evs = [f for f in evs if int(re.search(r"_v(\d+)\.tsv", f).group(1)) <= 12] or evs          # the matrix this step extends (v12)
EVp = pd.read_csv(evs[-1], sep="\t") if evs else pd.DataFrame(columns=["claim", "decision", "status", "key_numbers"])
naa = V.get("NAA") or []
ev = [dict(claim="External data (mouse hippocampus MALDI + Visium, another laboratory): NAA follows the neuronal synthesis side rather than the oligodendroglial "
                 "degradation side (dual-matched null)", decision="D64a", status=str(V.get("D64a_NAA_autonomy_external")),
           key_numbers="; ".join(f"{d.get('section')}: r_neu {d.get('neuronal_r', np.nan):+.2f} (P {d.get('neuronal_p_dual', np.nan):.3f}) "
                                 f"vs r_oli {d.get('oligodendroglial_r', np.nan):+.2f}" for d in naa)[:200]),
      dict(claim="External data: GEM own-pathway specificity is sparse", decision="D64b", status=str(V.get("D64b_sparse_specificity_external")),
           key_numbers=json.dumps({k: V.get("gem", {}).get(k) for k in ("n_tests", "frac_sig", "binom_p")}))]
vnum = int(re.search(r"_v(\d+)\.tsv", evs[-1]).group(1)) if evs else 12
EV = pd.concat([EVp, pd.DataFrame(ev)], ignore_index=True)
EV.to_csv(os.path.join(TAB, f"m23_evidence_matrix_v{vnum + 1}.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m23_evidence_matrix_v{vnum + 1}.tsv ({len(EV)} rows)")
