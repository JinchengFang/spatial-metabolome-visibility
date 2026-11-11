#!/usr/bin/env python3

import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step9")
T = lambda s: os.path.join(P["P2"], "05_tables", s)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), sep="\t") if os.path.exists(os.path.join(d, f)) else None
OS, PR, MO, CS_ = (rd(TAB, f) for f in ("m12_orthogonal_summary.tsv", "m12_conservation_pairs.tsv", "m12_modules.tsv", "m12_coupling_summary.tsv"))
EV8 = rd(T("Step8"), "m11_evidence_matrix.tsv")

verdict, ev = {}, []
if OS is not None:
    verdict["D37"] = f"{int(OS.D37_orthogonal_structured.sum())}/{len(OS)}"
    ev.append(dict(claim="Transcriptome-orthogonal metabolite variation is reliable and spatially structured", decision="D37", status=verdict["D37"],
                   key_numbers=f"median residual reliability {OS.residual_SB.median():.2f}; median fraction reliable and structured {OS.frac_orthogonal_reliable.median():.0%}"))
if PR is not None and len(PR):
    verdict["D38"] = f"{int((PR.p_residual < 0.05).sum())}/{len(PR)}"
    ev.append(dict(claim="Transcriptome-orthogonal metabolite covariation is conserved between patients", decision="D38", status=verdict["D38"] + " patient pairs",
                   key_numbers=f"median consistency of residual covariation rho {PR.rho_residual.median():.2f} (raw {PR.rho_raw.median():.2f})"))
if MO is not None and len(MO):
    verdict["D39"] = int(MO.conserved.sum())
    ev.append(dict(claim="Transcriptome-orthogonal metabolic modules exist that are conserved across patients and not technical", decision="D39",
                   status=f"{verdict['D39']} modules",
                   key_numbers="; ".join(f"{r.module} (n={r.size}, {r.classes[:40]})" for r in MO[MO.conserved].head(4).itertuples()) or "none"))
if CS_ is not None:
    r_ = CS_.set_index("variant")
    verdict["D36r"] = bool(r_.loc["axis-removed", "D36r_specific_coupling_reproducible"]) if "axis-removed" in r_.index else None
    ev.append(dict(claim="Transcriptomic coupling of metabolites (including shared tissue axes) is identifiable across patients", decision="D36r-raw",
                   status=str(bool(r_.loc["raw", "auc_median"] >= .65)) if "raw" in r_.index else "—",
                   key_numbers=f"median identification AUC {r_.loc['raw', 'auc_median']:.2f}" if "raw" in r_.index else ""))
    ev.append(dict(claim="After removing shared axes, metabolite-specific transcriptomic coupling remains identifiable", decision="D36r", status=str(verdict["D36r"]),
                   key_numbers=f"median identification AUC {r_.loc['axis-removed', 'auc_median']:.2f}" if "axis-removed" in r_.index else ""))
EV = pd.concat([EV8 if EV8 is not None else pd.DataFrame(), pd.DataFrame(ev)], ignore_index=True)
if len(EV):
    EV.to_csv(os.path.join(TAB, "m12_evidence_matrix_v2.tsv"), sep="\t", index=False)
C.save_json(verdict, os.path.join(TAB, "m12_step9_verdict.json"))
C.log(f"-> {TAB}/m12_evidence_matrix_v2.tsv ({len(EV)} rows), m12_step9_verdict.json")
