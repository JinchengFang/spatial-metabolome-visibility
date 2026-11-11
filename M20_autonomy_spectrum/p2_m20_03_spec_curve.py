#!/usr/bin/env python3

import argparse, itertools, json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import p2_common as C
from m20_helpers import make_mapper, AMINO
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--in-step", default="Step16"); ap.add_argument("--n-perm", type=int, default=999)
ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--step", default="Step17")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(17)
SM = pd.read_csv(os.path.join(P["P2"], "05_tables", a.in_step, "m20_sma_specificity.tsv"), sep="\t")
TOPO = pd.read_csv(os.path.join(P["PROC"], "m20_gem_topology.tsv"), sep="\t").set_index("met_id")
GEM = C.load_human_gem(P["REFDB"]); gem_ids, topo_of = make_mapper(GEM, C, TOPO)
T = pd.DataFrame([dict(metabolite=m, **topo_of(gem_ids(m))) for m in SM.metabolite.unique()]).set_index("metabolite")
den = T.n_export_gene + T.n_chem_gene + T.n_internal_gene
PRED = {"export_share": T.export_share, "log_export_count": np.log1p(T.n_export_gene), "export_binary": (T.export_share > 0.25).astype(float),
        "export_share_incl_internal": T.n_export_gene / den.clip(lower=1), "any_transport_share": (T.n_export_gene + T.n_internal_gene) / den.clip(lower=1)}
C.log(f"{SM.metabolite.nunique()} metabolites; {int(T.export_share.notna().sum())} mapped to the topology")


def perm_rho(x, y, Z=None):
    rx, ry = rankdata(x), rankdata(y)
    if Z is not None:
        D = np.column_stack([np.ones(len(rx))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])])
        rx = rx - D @ np.linalg.lstsq(D, rx, rcond=None)[0]; ry = ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1]); nul = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(a.n_perm)])
    return r, float((np.sum(nul <= r) + 1) / (a.n_perm + 1))


rows = []
for level in ("full", "detrended"):
    S = SM[SM.level == level]
    agg = S.groupby("metabolite").agg(pctl=("pctl", "mean"), z=("z", "mean"), n_sections=("pctl", "size"), max_ions=("n_ions", "max"), n_genes=("n_genes", "median"),
                                      matrices=("matrix", lambda s: ",".join(sorted(set(s)))))
    SUBSETS = {"all": lambda d: d, "high_conf_annotation": lambda d: d[d.max_ions >= 2], "drop_DHB": lambda d: d[d.matrices != "DHB"],
               "drop_9AA": lambda d: d[d.matrices != "9AA"], "drop_FMP10": lambda d: d[d.matrices != "FMP10"],
               "no_amino_acids": lambda d: d[~d.index.isin(AMINO)], "multi_section": lambda d: d[d.n_sections >= 2]}
    for (pn, pv), (sn, sf), oc, cv in itertools.product(PRED.items(), SUBSETS.items(), ("pctl", "z"), ("none", "setsize+degree")):
        if sn.startswith("drop_"):                                            # leave one matrix out: re-average over the sections of the other matrices
            mtx = sn.split("_")[1]; d = S[S.matrix != mtx].groupby("metabolite").agg(pctl=("pctl", "mean"), z=("z", "mean"), n_genes=("n_genes", "median"))
        else:
            d = sf(agg)
        d = d.join(pv.rename("x"), how="inner").join(T[["n_chem_gene"]], how="inner").dropna(subset=["x", oc])
        if len(d) < 10:
            continue
        Z = np.column_stack([np.log(d.n_genes.values), np.log1p(d.n_chem_gene.values)]) if cv != "none" else None
        r, p = perm_rho(d.x.values, d[oc].values, Z)
        rows.append(dict(level=level, predictor=pn, subset=sn, outcome=oc, covariates=cv, n=len(d), rho=r, p_one_sided=p))
SC = pd.DataFrame(rows); SC.to_csv(os.path.join(TAB, "m20_spec_curve.tsv"), sep="\t", index=False)
# baseline bootstrap
d0 = SM[SM.level == "full"].groupby("metabolite").pctl.mean().to_frame().join(T.export_share, how="inner").dropna()
bs = []; rb = np.random.default_rng(1717)                                   # independent random stream for the bootstrap
for _ in range(a.n_boot):
    i = rb.integers(0, len(d0), len(d0)); x, y = d0.export_share.values[i], d0.pctl.values[i]
    if np.std(x) > 0:
        bs.append(spearmanr(x, y).correlation)
ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
v = {}
for level in ("full", "detrended"):
    s = SC[SC.level == level]
    v[level] = dict(n_variants=len(s), frac_negative=float((s.rho < 0).mean()), frac_p05=float((s.p_one_sided < 0.05).mean()), median_rho=float(s.rho.median()),
                    by_subset=s.groupby("subset").rho.median().round(3).to_dict(), by_predictor=s.groupby("predictor").rho.median().round(3).to_dict())
v["base_rho"] = float(spearmanr(d0.export_share, d0.pctl).correlation); v["base_boot_ci95"] = ci; v["n_metabolites"] = len(d0)
v["D59b_robust"] = bool(v["full"]["frac_negative"] >= 0.8 and ci[1] < 0)
lomo = {k: v_ for k, v_ in v["full"]["by_subset"].items() if k.startswith("drop_")}
v["D59b_matrix_independent"] = bool(len(lomo) and max(lomo.values()) < 0)        # supplementary criterion: median rho < 0 after leaving out any one matrix
C.save_json(v, os.path.join(TAB, "m20_spec_verdict.json"))
C.log("specification curve (full, median rho by subset x predictor):\n" + SC[SC.level == "full"].pivot_table(index="subset", columns="predictor", values="rho", aggfunc="median").round(3).to_string()
      + "\ndecision: " + json.dumps(v, ensure_ascii=False, default=str))
fig, axs = plt.subplots(2, 1, figsize=(12, 5.6), sharex=False)
for ax_, level in zip(axs, ("full", "detrended")):
    s = SC[SC.level == level].sort_values("rho").reset_index(drop=True)
    ax_.scatter(np.arange(len(s)), s.rho, c=np.where(s.p_one_sided < 0.05, "#de2d26", "#bdbdbd"), s=8)
    ax_.axhline(0, c="k", lw=.5); ax_.set_ylabel("Spearman ρ"); ax_.set_xticks([])
    ax_.set_title(f"{level}: {len(s)} specifications, {100 * v[level]['frac_negative']:.0f}% negative, {100 * v[level]['frac_p05']:.0f}% P < 0.05 (red); "
                  f"base ρ {v['base_rho']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if level == "full" else f"{level}: {len(s)} specifications, {100 * v[level]['frac_negative']:.0f}% negative", fontsize=7)
axs[1].set_xlabel("specifications (ranked by ρ)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m20_spec_curve.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
