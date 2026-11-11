#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step16")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
V = json.load(open(os.path.join(TAB, "m20_verdict.json"))) if os.path.exists(os.path.join(TAB, "m20_verdict.json")) else {}
EV8 = rd(T("Step15"), "m19_evidence_matrix_v8.tsv")
mf, mh = V.get("mouse_full", {}), V.get("human", {})
ev = [dict(claim="Same section: the fraction of metabolites specifically predicted by their own GEM pathway genes exceeds chance", decision="D59a",
           status=str(V.get("D59a_specificity_exists_same_section")),
           key_numbers=f"{100 * mf.get('frac_sig', np.nan):.1f}% of {mf.get('n_tests')} tests (binomial P = {mf.get('binom_p', np.nan):.2g})"),
      dict(claim="Human-GEM topology (share of trans-membrane transport) predicts a priori which metabolites the transcriptome can see", decision="D59b",
           status=str(V.get("D59b_export_predicts_low_autonomy")),
           key_numbers=f"rho = {mf.get('rho', np.nan):+.2f}, P = {mf.get('p_perm', np.nan):.3f}, n = {mf.get('n_metabolites')}; "
                       f"adjusted for gene-set size {mf.get('partial_rho_given_setsize', np.nan):+.2f}"),
      dict(claim="Classical released transmitters are overall less visible", decision="D59c", status=str(V.get("D59c_released_lower")),
           key_numbers=f"{mf.get('released_mean_pctl', np.nan):.2f} vs {mf.get('retained_mean_pctl', np.nan):.2f} (MWU P = {mf.get('mwu_p', np.nan):.2f})"),
      dict(claim="Same-sign replication in human tumours (adjacent sections)", decision="D59d", status=str(V.get("D59d_human_same_sign")),
           key_numbers=f"rho = {mh.get('rho', np.nan):+.2f}, P = {mh.get('p_perm', np.nan):.3f}, n = {mh.get('n_metabolites')}")]
EV = pd.concat([EV8 if EV8 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
EV.to_csv(os.path.join(TAB, "m20_evidence_matrix_v9.tsv"), sep="\t", index=False)
C.log(f"-> {TAB}/m20_evidence_matrix_v9.tsv ({len(EV)} rows)")
