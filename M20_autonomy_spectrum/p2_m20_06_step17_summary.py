#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step17")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
rj = lambda f: json.load(open(os.path.join(TAB, f))) if os.path.exists(os.path.join(TAB, f)) else {}
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
SC, CP, ST = rj("m20_spec_verdict.json"), rj("m20_cptac_verdict.json"), rj("m20_strict_verdict.json")
EV9 = rd(os.path.join(P["P2"], "05_tables", "Step16"), "m20_evidence_matrix_v9.tsv")
ev = [dict(claim="The autonomy spectrum does not depend on a single matrix (supplementary criterion)", decision="D59b_matrix_independent",
           status=str(SC.get("D59b_matrix_independent")), key_numbers=json.dumps(SC.get("full", {}).get("by_subset", {}))[:160]),
      dict(claim="The autonomy spectrum (GEM topology predicts visibility) is robust across 140 specifications", decision="D59b_robust", status=str(SC.get("D59b_robust")),
           key_numbers=f"full level {100 * SC.get('full', {}).get('frac_negative', np.nan):.0f}% negative, {100 * SC.get('full', {}).get('frac_p05', np.nan):.0f}% P < 0.05; "
                       f"baseline rho {SC.get('base_rho', np.nan):+.2f} CI {SC.get('base_boot_ci95')}"),
      dict(claim="Own-pathway specificity exists at the population level (CPTAC)", decision="D60a", status=str(CP.get("D60a_population_specificity")),
           key_numbers=json.dumps(CP.get("full", {}))[:160]),
      dict(claim="At the population level, export share predicts specificity", decision="D60b", status=str(CP.get("D60b_export_predicts_low_specificity")),
           key_numbers=f"rho {CP.get('full', {}).get('rho', np.nan):+.2f}"),
      dict(claim="Dopamine soma vs terminal holds under the expression x spatial-structure matched null", decision="D54_strict", status=str(ST.get("D54_strict")),
           key_numbers=json.dumps(ST.get("dopamine"))),
      dict(claim="NAA-neuronal synthesis-side co-localization holds under the strict null", decision="D58a_strict", status=str(ST.get("D58a_strict")),
           key_numbers=json.dumps(ST.get("NAA"))),
      dict(claim="Same-section GEM own-pathway specificity remains above chance under the strict null", decision="D59a_strict", status=str(ST.get("D59a_strict")),
           key_numbers=json.dumps(ST.get("gem"))[:160])]
EV = pd.concat([EV9 if EV9 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
EV.to_csv(os.path.join(TAB, "m20_evidence_matrix_v10.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m20_evidence_matrix_v10.tsv ({len(EV)} rows)")
