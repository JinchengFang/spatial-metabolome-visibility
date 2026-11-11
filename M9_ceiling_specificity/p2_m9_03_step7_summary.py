#!/usr/bin/env python3

import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step7"); R0 = a.res
rd = lambda f: pd.read_csv(os.path.join(TAB, f), sep="\t") if os.path.exists(os.path.join(TAB, f)) else None
CS, GS, SS = rd("m9_ceiling_summary.tsv"), rd("m9_gem_summary.tsv"), rd("m7_shapley_summary.tsv")


def frac_true(df, col):
    if df is None or col not in df or not len(df):
        return "—"
    return f"{int(df[col].sum())}/{len(df)}"


verdict = {}
if CS is not None:
    for (ds, lv), g in CS[CS.res_um == R0].groupby(["dataset", "level"]):
        verdict[f"{ds}|{lv}"] = dict(n_samples=len(g), D30=frac_true(g, "D30_gap_not_noise"), D31=frac_true(g, "D31_depth_not_limiting"),
                                     D32=frac_true(g, "D32_internal_exceeds_tx"), SB=float(g.SB.median()), R2_tx=float(g.R2_tx_full.median()),
                                     tx_norm=float(g.tx_norm_reliable.median()), maldi_cross=float(g.R2_maldi_cross.median()))
if GS is not None:
    for (ds, lv), g in GS[GS.res_um == R0].groupby(["dataset", "level"]):
        verdict.setdefault(f"{ds}|{lv}", {}).update(dict(D33=frac_true(g, "D33_no_specificity"), gem_pctl=float(g.pctl_med_gem.median()),
                                                         gem_frac_p05=float(g.frac_p05_gem.median())))
if SS is not None:
    for (ds, lv), g in SS[SS.res_um == R0].groupby(["dataset", "level"]):
        verdict.setdefault(f"{ds}|{lv}", {}).update(dict(D29=frac_true(g, "D29_state_encoded"), frac_meaningful_state=float(g.frac_meaningful_state.median()),
                                                         frac_meaningful_resid=float(g.frac_meaningful_resid.median()) if "frac_meaningful_resid" in g else np.nan))
C.save_json(verdict, os.path.join(TAB, "m9_step7_verdict.json"))
C.log(f"-> {TAB}/m9_step7_verdict.json\n" + pd.DataFrame(verdict).T.to_string())
