#!/usr/bin/env python3

import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import p2_common as C
from m20_helpers import make_mapper, batch_X, batch_cv_r2, matched_sets, AMINO
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--n-null", type=int, default=499); ap.add_argument("--n-perm", type=int, default=9999)
ap.add_argument("--n-rep", type=int, default=5); ap.add_argument("--n-pc", type=int, default=3); ap.add_argument("--step", default="Step17")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(60)
dcp = os.path.join(P["PROC"], "cptac")
met = pd.read_parquet(os.path.join(dcp, "metabolome_lcimputed.parquet")); rna = pd.read_parquet(os.path.join(dcp, "rna.parquet"))
prog = pd.read_csv(os.path.join(dcp, "programs_rna.tsv"), sep="\t", index_col=0)
com = met.index.intersection(rna.index).intersection(prog.index); met, rna, prog = met.loc[com], rna.loc[com], prog.loc[com]
X = rna.astype(float); X = np.log2(X + 1) if np.nanmax(X.values) > 50 else X
X = X.loc[:, (X.notna().mean() >= 0.9) & (X.std() > 0)]; X = X.fillna(X.mean())
TOPO = pd.read_csv(os.path.join(P["PROC"], "m20_gem_topology.tsv"), sep="\t").set_index("met_id")
GEM = C.load_human_gem(P["REFDB"]); GEM_GENES = set().union(*GEM["met_genes"].values()); gem_ids, topo_of = make_mapper(GEM, C, TOPO)
REV = {v_: k for k, v_ in C.CPTAC_NAME.items()}
genes = np.asarray(X.columns).astype(str); gidx = {g: i for i, g in enumerate(genes)}
dec = np.minimum((pd.Series(X.mean().values).rank(pct=True).values * 10).astype(int), 9); U = np.array([gidx[g] for g in genes if g in GEM_GENES], int)
comp_cols = [c for c in ("Neuron_cortex", "WM_oligodendrocyte", "Reactive_astro", "Microglia", "Macrophage_TAM", "Vascular") if c in prog]
MAL = [p for p in ("AC_like", "OPC_like", "NPC_like", "MES1_inflammatory", "Hypoxia_MES2", "Cycling") if p in prog]
Zc = np.column_stack([np.ones(len(com))] + [C.zcols(prog[[c]].values)[:, 0] for c in comp_cols] + ([C.zcols(prog[MAL].values).mean(1)] if MAL else []))
resid = lambda M: M - Zc @ np.linalg.lstsq(Zc, M, rcond=None)[0]
G_full = np.nan_to_num(C.zcols(X.values)); G_bc = np.nan_to_num(C.zcols(resid(X.values)))
FOLDS = [rng.permutation(np.arange(len(com)) % 5) for _ in range(a.n_rep)]
C.log(f"CPTAC: {len(com)} tumours, {X.shape[1]} genes ({len(U)} GEM metabolic genes), {met.shape[1]} metabolites; composition covariates {comp_cols + (['malignant'] if MAL else [])}")
rows = []
for col in met.columns:
    if str(col).startswith("Unknown"):
        continue
    nm = REV.get(col, col); ids = gem_ids(nm) or gem_ids(col)
    if not ids:
        continue
    own_g = set().union(*[GEM["met_genes"].get(i, set()) for i in ids]); own = np.array(sorted(gidx[g] for g in own_g if g in gidx), int)
    if not (2 <= len(own) <= 150):
        continue
    y0 = met[col].values.astype(float)
    if np.std(y0) == 0:
        continue
    sets = [own] + matched_sets(own, U, dec, a.n_null, rng); tp = topo_of(ids)
    for level, G in (("full", G_full), ("beyond_comp", G_bc)):
        y = C.zcols((y0 if level == "full" else resid(y0[:, None])[:, 0])[:, None])[:, 0]
        Xb = batch_X(G, sets, a.n_pc); r_ = np.mean([batch_cv_r2(Xb, y, f) for f in FOLDS], axis=0); nul = r_[1:]
        rows.append(dict(level=level, metabolite=col, lib_name=nm, n_genes=len(own), genes=";".join(genes[own][:15]), R2_own=float(r_[0]), null_med=float(np.median(nul)),
                         pctl=float(np.mean(nul < r_[0])), p=float((np.sum(nul >= r_[0]) + 1) / (len(nul) + 1)), amino_acid=nm in AMINO or str(col).lower().lstrip("dl-") in AMINO, **tp))
D = pd.DataFrame(rows); D.to_csv(os.path.join(TAB, "m20_cptac_autonomy.tsv"), sep="\t", index=False)


def perm_rho(x, y):
    rx, ry = rankdata(x), rankdata(y); r = float(np.corrcoef(rx, ry)[0, 1])
    nul = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(a.n_perm)]); return r, float((np.sum(nul <= r) + 1) / (a.n_perm + 1))


v = {}
for level in ("full", "beyond_comp"):
    d = D[D.level == level].dropna(subset=["export_share"]); out = dict(n=len(d), frac_sig=float((d.p < 0.05).mean()) if len(d) else np.nan)
    if len(d):
        out["binom_p"] = float(binomtest(int((d.p < 0.05).sum()), len(d), 0.05, alternative="greater").pvalue)
    if len(d) >= 15:
        out["rho"], out["p_perm"] = perm_rho(d.export_share.values, d.pctl.values)
        dn = d[~d.amino_acid]
        if len(dn) >= 10:
            out["rho_no_amino"], out["p_no_amino"] = perm_rho(dn.export_share.values, dn.pctl.values)
    v[level] = out
f = v.get("full", {})
v["D60a_population_specificity"] = bool(f.get("binom_p", 1) < 0.05)
v["D60b_export_predicts_low_specificity"] = bool(f.get("n", 0) >= 15 and f.get("rho", 0) < 0 and f.get("p_perm", 1) < 0.05)
v["controls"] = D[(D.level == "full") & D.metabolite.str.contains("aspartic|hydroxyglutar", case=False)][["metabolite", "n_genes", "pctl", "p", "export_share"]].round(3).to_dict("records")
C.save_json(v, os.path.join(TAB, "m20_cptac_verdict.json"))
C.log("CPTAC (full):\n" + D[D.level == "full"].sort_values("pctl", ascending=False)[["metabolite", "n_genes", "R2_own", "null_med", "pctl", "p", "export_share"]].round(3).to_string(index=False)
      + "\ndecision: " + json.dumps(v, ensure_ascii=False, default=str))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax_, level in zip(axs, ("full", "beyond_comp")):
    d = D[D.level == level].dropna(subset=["export_share"])
    ax_.scatter(d.export_share, d.pctl, c=np.where(d.p < 0.05, "#de2d26", "#3182bd"), s=14, edgecolor="k", lw=.3)
    for r_ in d.itertuples():
        if r_.p < 0.05 or "aspartic" in r_.metabolite.lower():
            ax_.text(r_.export_share, r_.pctl, " " + str(r_.metabolite)[:22], fontsize=5)
    ax_.axhline(0.5, c="grey", lw=.5, ls="--"); ax_.set_xlabel("export share (Human-GEM)"); ax_.set_ylabel("own-pathway specificity across patients (percentile)")
    ax_.set_title(f"CPTAC GBM, {level}: ρ = {v[level].get('rho', np.nan):+.2f}, P = {v[level].get('p_perm', np.nan):.3f}; {100 * v[level].get('frac_sig', np.nan):.0f}% P<0.05", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m20_cptac_autonomy.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
