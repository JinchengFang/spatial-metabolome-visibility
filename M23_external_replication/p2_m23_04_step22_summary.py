#!/usr/bin/env python3

import argparse, glob, os, re, sys, json
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step22")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
V = json.load(open(os.path.join(TAB, "m23x_verdict.json"))) if os.path.exists(os.path.join(TAB, "m23x_verdict.json")) else {}
S = pd.read_csv(os.path.join(TAB, "m23x_summary.tsv"), sep="\t") if os.path.exists(os.path.join(TAB, "m23x_summary.tsv")) else pd.DataFrame()
evs = sorted(glob.glob(os.path.join(P["P2"], "05_tables", "Step*", "*evidence_matrix_v*.tsv")), key=lambda f: int(re.search(r"_v(\d+)\.tsv", f).group(1)))
evs = [f for f in evs if int(re.search(r"_v(\d+)\.tsv", f).group(1)) <= 13] or evs          # the matrix this step extends (v13)
EVp = pd.read_csv(evs[-1], sep="\t") if evs else pd.DataFrame(columns=["claim", "decision", "status", "key_numbers"]); vnum = int(re.search(r"_v(\d+)\.tsv", evs[-1]).group(1)) if evs else 13
EVp = EVp[~EVp.decision.astype(str).isin(["D64a", "D64b", "D65a", "D65b", "D65c"])]
s2 = S[S.res_um == 200] if len(S) else S
kn = lambda col, fmt: "; ".join(f"{r.section} {fmt.format(getattr(r, col))}" for r in s2.itertuples()) if len(s2) else ""
ev = [dict(claim="External data (mouse hippocampus MALDI + Visium, another laboratory): NAA autonomy", decision="D64a", status="not testable",
           key_numbers="the public matrices provide only feature indices (V3...V1550) without m/z, so features cannot be annotated"),
      dict(claim="External data: GEM own-pathway specificity is sparse", decision="D64b", status="not testable", key_numbers="as D64a"),
      dict(claim="External data: the transcriptome reaches only part of the reliable metabolite ceiling (median R²_tx/SB < 0.75)", decision="D65a",
           status=str(V.get("D65a_tx_partial")), key_numbers=kn("ratio_median", "{:.2f}") + "; Vicari same section 0.54"),
      dict(claim="External data: transcriptome-orthogonal variation is reproducible (residual split-half r > 0.3)", decision="D65b",
           status=str(V.get("D65b_orthogonal_reproducible")), key_numbers=kn("r_resid_median", "{:.2f}")),
      dict(claim="External data: internal (cross-half) metabolome predictability exceeds that of the transcriptome", decision="D65c",
           status=str(V.get("D65c_maldi_internal_gt_tx")),
           key_numbers="; ".join(f"{r.section} {r.R2_maldi_median:.2f} vs {r.R2_tx_median:.2f}" for r in s2.itertuples()) if len(s2) else "")]
EV = pd.concat([EVp, pd.DataFrame(ev)], ignore_index=True); evname = f"m23_evidence_matrix_v{vnum + 1}.tsv"; EV.to_csv(os.path.join(TAB, evname), sep="\t", index=False)
C.log(f"-> {TAB}/{evname} ({len(EV)} rows)")
