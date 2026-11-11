#!/usr/bin/env python3

import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step6")
S = pd.read_csv(os.path.join(TAB, "m7_shapley_summary.tsv"), sep="\t")
verdict = {}
for (ds, lv), d in S[S.res_um == a.res].groupby(["dataset", "level"]):
    verdict[f"{ds}|{lv}"] = dict(n=len(d), D27=int(d.D27_comp_ge_state.sum()), D28=int(d.D28_tech_ge_20pct.sum()), D29=int(d.D29_state_encoded.sum()),
                                 R2_full=float(d.R2_full.median()), R2_other=float(d.R2_other.median()), unique_state=float(d.unique_state_adj.median()))
C.save_json(verdict, os.path.join(TAB, "m7_step6_verdict.json"))
C.log(f"-> {TAB}/m7_step6_verdict.json\n" + pd.DataFrame(verdict).T.round(3).to_string())
