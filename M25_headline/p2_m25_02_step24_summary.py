#!/usr/bin/env python3

import argparse, glob, json, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--step", default="Step24")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
fh = os.path.join(TAB, "m25_headline.json")
if not os.path.exists(fh):
    raise SystemExit("Missing m25_headline.json: run M25-01 first")
HJ = json.load(open(fh))
evs = sorted(glob.glob(os.path.join(P["P2"], "05_tables", "Step*", "*evidence_matrix_v*.tsv")), key=lambda f: int(re.search(r"_v(\d+)\.tsv", f).group(1)))
evs = [f for f in evs if int(re.search(r"_v(\d+)\.tsv", f).group(1)) <= 15] or evs          # the matrix this step extends (v15)
EVp = pd.read_csv(evs[-1], sep="\t"); vnum = int(re.search(r"_v(\d+)\.tsv", evs[-1]).group(1)); EVp = EVp[~EVp.decision.astype(str).isin(["D68a", "D68b"])]
G = lambda g: HJ.get(g, {})
ci = lambda g: (f"{G(g)['ci'][0]:.2f}–{G(g)['ci'][1]:.2f}" if G(g).get("ci") and G(g)["ci"][0] is not None else "NA")
dp = HJ.get("design_pairs", {})
psd = "; ".join(f"{k.replace('v11l12-', '')} {v:.3f}".replace(" -", " −") for k, v in dp.get("per_maldi_section_diff", {}).items())
ev = [dict(claim="Within the same mouse (tissue / animal / matrix / peak list unchanged), adjacent-section pairing gives a lower calibrated fraction than "
                 "same-section pairing (≥ 5 of 6 pairs and all 3 MALDI sections)", decision="D68a", status=str(HJ.get("D68a_within_mouse_design_penalty")),
           key_numbers=f"{dp.get('n_lower', '?')}/{dp.get('n_pairs', '?')} pairs lower; median retained fraction of same-section value {dp.get('median_retained', '?')}; "
                       f"difference by MALDI section: {psd}"),
      dict(claim="Main conclusion: upper 95% cluster-bootstrap CI of the calibrated fraction over all real designs < 0.75", decision="D68b",
           status=str(HJ.get("D68b_headline_upper_CI_below_0.75")),
           key_numbers=f"All {G('all real designs').get('calibrated')} [{ci('all real designs')}]; same section {G('same-section (mouse brain, Vicari)').get('calibrated')} "
                       f"[{ci('same-section (mouse brain, Vicari)')}]; external {G('external (mouse hippocampus, another lab)').get('calibrated')}; "
                       f"adjacent-section human tumours {G('adjacent-section (human tumours)').get('calibrated')} [{ci('adjacent-section (human tumours)')}]")]
EV = pd.concat([EVp, pd.DataFrame(ev)], ignore_index=True); evname = f"m25_evidence_matrix_v{vnum + 1}.tsv"; EV.to_csv(os.path.join(TAB, evname), sep="\t", index=False)
st = EV.status.astype(str).str.strip()
C.log(f"-> {TAB}/{evname} ({len(EV)} rows; {int((st == 'True').sum())} supported, {int((st == 'False').sum())} not supported)")
