#!/usr/bin/env python3

import argparse, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--force", action="store_true")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step3")
SUP = os.path.join(P["CPTAC"], "Wang2021_CancerCell_supp"); OUT = os.path.join(P["PROC"], "cptac"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7})
f2, f3 = os.path.join(SUP, "mmc2.xlsx"), os.path.join(SUP, "mmc3.xlsx")
if not (os.path.exists(f2) and os.path.exists(f3)):
    raise SystemExit(f"{f2} or {f3} missing (Wang et al. 2021 Cancer Cell, Tables S1 / S2)")

SHEETS = {"rna": "gene_expression_fpkm_uq", "protein": "proteome_normalized", "metabolome": "metabolome_normalized",
          "lipid_pos": "lipidome_positive_normalized", "lipid_neg": "lipidome_negative_normalized"}
cache = {k: os.path.join(OUT, f"{k}.parquet") for k in SHEETS}
if a.force or not all(os.path.exists(v) for v in cache.values()):
    t0 = time.time(); C.log("reading mmc3.xlsx (about 130 MB, once; takes a few minutes) ...")
    raw = pd.read_excel(f3, sheet_name=list(SHEETS.values()))
    C.log(f"  read in {time.time() - t0:.0f} s")
    r = raw[SHEETS["rna"]]
    r = r[r.gene_type == "protein_coding"]
    cases = [c for c in r.columns if str(c).startswith(("C3L", "C3N"))]
    E = r[["gene_name"] + cases].copy()
    E[cases] = np.log2(E[cases].apply(pd.to_numeric, errors="coerce") + 1)
    E["_m"] = E[cases].mean(axis=1); E = E.sort_values("_m", ascending=False).drop_duplicates("gene_name").drop(columns="_m")
    E.set_index("gene_name").T.to_parquet(cache["rna"])
    p = raw[SHEETS["protein"]]; cases = [c for c in p.columns if str(c).startswith(("C3L", "C3N"))]
    p = p.drop_duplicates("symbol").set_index("symbol")[cases].apply(pd.to_numeric, errors="coerce"); p.T.to_parquet(cache["protein"])
    for k, col in (("metabolome", "Metabolite"), ("lipid_pos", "Lipid"), ("lipid_neg", "Lipid")):
        m = raw[SHEETS[k]]; cases = [c for c in m.columns if str(c).startswith(("C3L", "C3N"))]
        m = m.drop_duplicates(col).set_index(col)[cases].apply(pd.to_numeric, errors="coerce"); m.T.to_parquet(cache[k])
D = {k: pd.read_parquet(v) for k, v in cache.items()}
ann = pd.read_excel(f2, sheet_name="additional_annotations").set_index("case")
cli = pd.read_excel(f2, sheet_name="clinical_data").set_index("case_id")
ann = ann[ann.sample_type.astype(str).str.lower() == "tumor"] if "sample_type" in ann else ann
excl = cli.index[cli.get("excluded_in_downstream_analysis", pd.Series(0, index=cli.index)).fillna(0).astype(str).isin(["1", "1.0", "True", "Yes"])]
ann = ann.drop(index=[c for c in excl if c in ann.index])
ann["IDH1_mut"] = ann.IDH1_alter_status.notna().astype(int)
ann = ann.join(cli[[c for c in ("age", "gender", "vital_status", "path_diag_to_death_days", "path_diag_to_last_contact_days") if c in cli]], how="left")
for k in D:
    D[k] = D[k].loc[[i for i in D[k].index if i in ann.index]]
ann.to_csv(os.path.join(OUT, "annotations.tsv"), sep="\t")
ov = pd.DataFrame({k: [len(v), v.shape[1]] for k, v in D.items()}, index=["n_tumors", "n_features"]).T
ov["n_with_rna_and_this"] = [len(set(v.index) & set(D["rna"].index)) for v in D.values()]
ov.to_csv(os.path.join(TAB, "cptac_sample_overlap.tsv"), sep="\t")
C.log("samples / features:\n" + ov.to_string())
C.log(f"IDH1-mutant {int(ann.IDH1_mut.sum())}; G-CIMP {int(ann.is_gcimp.fillna(0).sum())}; {len(ann)} tumours")

# ---------------------------------------------------------------- programme scores
for mod in ("rna", "protein"):
    X = D[mod]
    Xv = X.fillna(X.mean()).values
    S, used = C.score_programs(Xv, X.columns, {**C.PROGRAMS, **C.METAB_PROGRAMS}, min_genes=3)
    S.index = X.index
    S.to_csv(os.path.join(OUT, f"programs_{mod}.tsv"), sep="\t")
    C.log(f"  {mod} programmes: {S.shape[1]} (genes used per programme: {', '.join(f'{k}:{len(v)}' for k, v in list(used.items())[:6])} ...)")
PR = pd.read_csv(os.path.join(OUT, "programs_rna.tsv"), sep="\t", index_col=0)
# GC-MS missing values are mostly below the detection limit (left-censored): imputed with minimum - 1 (log2 scale = half the minimum); detection rate recorded
for k in ("metabolome", "lipid_pos", "lipid_neg"):
    X = D[k]; det = X.notna().mean()
    Xi = X.apply(lambda c: c.fillna(c.min() - 1.0) if c.notna().any() else c)
    Xi.to_parquet(os.path.join(OUT, f"{k}_lcimputed.parquet")); det.rename("detection_rate").to_frame().to_csv(os.path.join(OUT, f"{k}_detection.tsv"), sep="\t")
    C.log(f"  {k}: median detection rate {det.median():.2f}; {int((det < .25).sum())} features detected in < 25% (correlations use features detected in >= 20 tumours)")
M = pd.read_parquet(os.path.join(OUT, "metabolome_lcimputed.parquet")); Mdet = D["metabolome"].notna()

# ---------------------------------------------------------------- positive controls
rows = []
hg = "2-hydroxyglutaric acid"
if hg in M:
    for grp in ("IDH1_mut", "is_gcimp"):
        g = ann.reindex(M.index)[grp].fillna(0).astype(int)
        x1, x0 = M.loc[g == 1, hg].dropna(), M.loc[g == 0, hg].dropna()
        if len(x1) >= 2:
            u = mannwhitneyu(x1, x0, alternative="greater")
            delta = 2 * u.statistic / (len(x1) * len(x0)) - 1
            rows.append(dict(control=f"2-HG higher in {grp}", n1=len(x1), n0=len(x0), effect=f"Cliff δ = {delta:.2f}", stat=delta, p=u.pvalue,
                             log2FC=float(x1.mean() - x0.mean())))
for met, prog in (("N-acetyl-L-aspartic acid", "Neuron_cortex"), ("lactic acid", "Hypoxia_MES2"), ("lactic acid", "Glycolysis"),
                  ("2-hydroxyglutaric acid", "Hypoxia_MES2")):
    if met in M and prog in PR:
        ix = M.index.intersection(PR.index); nd = int(Mdet.loc[ix, met].sum())
        if nd >= 20:
            r, pv, n = C.partial_spearman(M.loc[ix, met], PR.loc[ix, prog])
            rows.append(dict(control=f"{met} ~ {prog} (RNA program)", n1=n, effect=f"ρ = {r:.2f}", stat=r, p=pv))
        else:   # sparse metabolite: programme score in tumours with vs without detection
            d1, d0 = PR.loc[ix][Mdet.loc[ix, met]][prog], PR.loc[ix][~Mdet.loc[ix, met]][prog]
            u = mannwhitneyu(d1, d0, alternative="greater"); delta = 2 * u.statistic / (len(d1) * len(d0)) - 1
            rows.append(dict(control=f"{prog} higher where {met} detected ({nd}/{len(ix)})", n1=len(d1), n0=len(d0), effect=f"Cliff δ = {delta:.2f}", stat=delta, p=u.pvalue))
CT = pd.DataFrame(rows); CT.to_csv(os.path.join(TAB, "cptac_positive_controls.tsv"), sep="\t", index=False)
for r in CT.itertuples():
    C.log(f"  positive control {r.control}: {r.effect}, p = {r.p:.2g}")

fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
if hg in M:
    g = ann.reindex(M.index).IDH1_mut.fillna(0).astype(int)
    axs[0].boxplot([M.loc[g == 0, hg].dropna(), M.loc[g == 1, hg].dropna()]); axs[0].set_xticks([1, 2]); axs[0].set_xticklabels(["IDH1-wt", "IDH1-mut"])
    axs[0].set_ylabel("2-HG (log2, normalized)")
    for k_, xs in enumerate((M.loc[g == 0, hg].dropna(), M.loc[g == 1, hg].dropna()), 1):
        axs[0].scatter(np.full(len(xs), k_) + np.random.uniform(-.08, .08, len(xs)), xs, s=8, alpha=.6)
    axs[0].set_title("CPTAC: 2-HG by IDH1 status")
for axx, (met, prog) in zip(axs[1:], (("N-acetyl-L-aspartic acid", "Neuron_cortex"), ("lactic acid", "Hypoxia_MES2"))):
    if met in M and prog in PR:
        ix = M.index.intersection(PR.index); axx.scatter(PR.loc[ix, prog], M.loc[ix, met], s=10, alpha=.7)
        axx.set_xlabel(f"{prog} (RNA program)"); axx.set_ylabel(met); axx.set_title(f"n = {len(ix)}")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cptac_positive_controls.png"), dpi=150); plt.close(fig)
C.log(f"→ {OUT}")
