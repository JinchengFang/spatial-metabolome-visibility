#!/usr/bin/env python3

import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--min-detect", type=int, default=20)
ap.add_argument("--step", default="Step3", help="read / write 05_tables/<step> (the M4-01 export is also read from there)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); OUT = os.path.join(P["PROC"], "cptac")
plt.rcParams.update({"font.size": 7})
need = ["metabolome_lcimputed.parquet", "metabolome.parquet", "programs_rna.tsv", "programs_protein.tsv", "annotations.tsv"]
miss = [f for f in need if not os.path.exists(os.path.join(OUT, f))]
if miss:
    raise SystemExit(f"{miss} missing; run M8-01 first")
M = pd.read_parquet(os.path.join(OUT, "metabolome_lcimputed.parquet")); Mdet = pd.read_parquet(os.path.join(OUT, "metabolome.parquet")).notna()
ann = pd.read_csv(os.path.join(OUT, "annotations.tsv"), sep="\t", index_col=0)
PRG = {mod: pd.read_csv(os.path.join(OUT, f"programs_{mod}.tsv"), sep="\t", index_col=0) for mod in ("rna", "protein")}
named = [c for c in M.columns if not c.startswith("Unknown")]
CONF = ["Neuron_cortex", "WM_oligodendrocyte"]


def atlas(mod, ids):
    PR = PRG[mod]; ix = M.index.intersection(PR.index).intersection(ids)
    det = Mdet.loc[ix].sum(); mets = [m for m in named if det.get(m, 0) >= a.min_detect]
    rows = []
    for m in mets:
        for pg in PR.columns:
            r0, p0, n = C.partial_spearman(M.loc[ix, m], PR.loc[ix, pg])
            cv = [c for c in CONF if c in PR and c != pg]
            Z = np.column_stack([PR.loc[ix, cv].values, ann.reindex(ix).xcell_microenvironment_score.values])
            r1, p1, _ = C.partial_spearman(M.loc[ix, m], PR.loc[ix, pg], Z)
            rows.append(dict(modality=mod, metabolite=m, program=pg, n=n, n_detected=int(det[m]), rho=r0, p=p0, rho_adj=r1, p_adj=p1))
    T = pd.DataFrame(rows)
    if len(T):
        T["q"] = C.bh(T.p.values); T["q_adj"] = C.bh(T.p_adj.fillna(1).values)
    return T, ix


wt_ids = ann.index[ann.IDH1_mut.fillna(0) == 0]
AT = {}
for mod in ("rna", "protein"):
    T, ix = atlas(mod, ann.index); AT[mod] = T
    T.to_csv(os.path.join(TAB, f"m8_cptac_atlas_{mod}.tsv"), sep="\t", index=False)
    C.log(f"[{mod}] {len(ix)} tumours, {T.metabolite.nunique()} metabolites x {T.program.nunique()} programmes: "
          f"q < 0.05 raw {int((T.q < 0.05).sum())}, adjusted for normal brain / microenvironment {int((T.q_adj < 0.05).sum())}")
    for r in T.sort_values("p_adj").head(8).itertuples():
        C.log(f"     {r.metabolite:28s} ~ {r.program:24s} rho={r.rho:+.2f} adjusted rho={r.rho_adj:+.2f} q_adj={r.q_adj:.2g}")

# ------------------------------------------------------------------ B/C/D replication
f4 = os.path.join(TAB, "m4_couplings_for_cptac.tsv")
REP, summ = pd.DataFrame(), {}
if os.path.exists(f4):
    S4 = pd.read_csv(f4, sep="\t"); S4 = S4[S4.cptac_name.isin(M.columns)]
    rows = []
    for subset, ids in (("all", ann.index), ("IDH1_wt", wt_ids)):
        for mod in ("rna", "protein"):
            PR = PRG[mod]; ix = M.index.intersection(PR.index).intersection(ids)
            for r in S4.itertuples():
                if r.program not in PR:
                    continue
                cm = r.cptac_name; nd = int(Mdet.loc[ix, cm].sum()); sgn = np.sign(r.rho_spatial)
                if nd >= a.min_detect:
                    rho, p2, n = C.partial_spearman(M.loc[ix, cm], PR.loc[ix, r.program])
                    p1 = p2 / 2 if np.sign(rho) == sgn else 1 - p2 / 2; test = "Spearman"; eff = rho
                elif nd >= 5 and len(ix) - nd >= 5:
                    d1, d0 = PR.loc[ix][Mdet.loc[ix, cm]][r.program], PR.loc[ix][~Mdet.loc[ix, cm]][r.program]
                    u = mannwhitneyu(d1, d0, alternative="greater" if sgn > 0 else "less")
                    eff = 2 * mannwhitneyu(d1, d0).statistic / (len(d1) * len(d0)) - 1; p1 = u.pvalue; test = f"detected vs not ({nd}/{len(ix)})"; n = len(ix)
                else:
                    continue
                rows.append(dict(subset=subset, modality=mod, metabolite=r.metabolite, cptac_name=cm, program=r.program, spatial_feature=r.feature,
                                 rho_spatial=r.rho_spatial, q_spatial=r.q_spatial, spatial_robust=bool(r.robust),
                                 spatial_comp_indep=bool(r.composition_independent), test=test, effect_cptac=eff, p_one_sided=p1, n=n,
                                 same_sign=bool(np.sign(eff) == sgn)))
    REP = pd.DataFrame(rows)
    if len(REP):
        for key, g in REP[REP.spatial_robust].groupby(["subset", "modality"]):
            REP.loc[g.index, "q_rep"] = C.bh(g.p_one_sided.values)
        REP["replicated"] = REP.spatial_robust & REP.same_sign & (REP.p_one_sided < 0.05)
        REP["replicated_strong"] = REP.replicated & (REP.q_rep < 0.1)
        REP.to_csv(os.path.join(TAB, "m8_replication.tsv"), sep="\t", index=False)
        for (sub, mod), g in REP.groupby(["subset", "modality"]):
            gr = g[g.spatial_robust]; k = int(gr.same_sign.sum()); n_ = len(gr)
            bt = binomtest(k, n_, 0.5, alternative="greater").pvalue if n_ else np.nan
            sp_r = spearmanr(g.rho_spatial, g.effect_cptac) if len(g) >= 5 else (np.nan, np.nan)
            # sign agreement stratified by spatial effect size (are strong couplings replicated more often?)
            strat = {}
            for lo_, hi_ in ((0.0, 0.3), (0.3, 0.5), (0.5, 1.01)):
                h_ = g[(g.rho_spatial.abs() >= lo_) & (g.rho_spatial.abs() < hi_)]
                strat[f"|rho|{lo_}-{min(hi_, 1)}"] = dict(n=len(h_), same_sign=int(h_.same_sign.sum()),
                                                         binom_p=float(binomtest(int(h_.same_sign.sum()), len(h_), .5, alternative="greater").pvalue) if len(h_) else None)
            strat_msg = "     sign agreement by spatial |rho|: " + "; ".join(f"{k_} {v_['same_sign']}/{v_['n']}" + (f" (p={v_['binom_p']:.2g})" if v_['binom_p'] is not None else "") for k_, v_ in strat.items())
            summ[f"{sub}_{mod}"] = dict(sign_by_spatial_strength=strat, n_mappable=len(g), n_spatially_robust=n_, robust_same_sign=k, sign_binom_p=bt,
                                        replicated=int(gr.replicated.sum()), replicated_strong=int(gr.replicated_strong.sum()),
                                        rho_spatial_vs_cptac=float(sp_r[0]), p_rho=float(sp_r[1]),
                                        replicated_pairs=[f"{x.metabolite}~{x.program}" for x in gr[gr.replicated].itertuples()])
            C.log(f"[replication {sub} / {mod}] {n_} spatially robust pairs: same sign {k} (binomial p={bt:.2g}), replicated (one-sided p < 0.05) {int(gr.replicated.sum())}, "
                  f"BH q < 0.1 {int(gr.replicated_strong.sum())}; all {len(g)} pairs: Spearman of rho_spatial vs effect = {sp_r[0]:.2f} (p={sp_r[1]:.2g})")
            C.log(strat_msg)
        for x in REP[(REP.subset == "all") & (REP.modality == "rna") & REP.replicated].sort_values("p_one_sided").head(15).itertuples():
            C.log(f"     replicated: {x.metabolite:20s} ~ {x.program:22s} spatial rho={x.rho_spatial:+.2f} -> CPTAC {x.test} {x.effect_cptac:+.2f} p1={x.p_one_sided:.2g}")
else:
    C.log("m4_couplings_for_cptac.tsv of M4-01 not found; replication skipped (run M4-01 first)")
C.save_json(summ, os.path.join(TAB, "m8_replication_summary.json"))

# ------------------------------------------------------------------ figures
T = AT["rna"]
if len(T):
    H = T.pivot(index="metabolite", columns="program", values="rho_adj"); Q = T.pivot(index="metabolite", columns="program", values="q_adj")
    keep = H.index[(Q < 0.1).any(axis=1)] if (Q < 0.1).any(axis=None) else H.abs().max(1).sort_values().index[-20:]
    H, Q = H.loc[keep], Q.loc[keep]
    fig, ax = plt.subplots(figsize=(.3 * H.shape[1] + 3, .22 * H.shape[0] + 1.6))
    im = ax.imshow(H.values, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if Q.values[i, j] < 0.1:
                ax.text(j, i, "*" if Q.values[i, j] >= 0.05 else "**", ha="center", va="center", fontsize=6)
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index); ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=90)
    plt.colorbar(im, ax=ax, shrink=.6, label="partial Spearman ρ (adj. normal brain + xCell)")
    ax.set_title("CPTAC GBM: metabolite × RNA program (* q<0.1, ** q<0.05)"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "m8_cptac_atlas_heatmap.png"), dpi=160); plt.close(fig)
if len(REP):
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.3))
    for axx, mod in zip(axs, ("rna", "protein")):
        g = REP[(REP.subset == "all") & (REP.modality == mod)]
        axx.scatter(g.rho_spatial, g.effect_cptac, c=np.where(g.replicated, "crimson", np.where(g.spatial_robust, "grey", "lightgrey")), s=12)
        for x in g[g.replicated].itertuples():
            axx.annotate(f"{x.metabolite}~{x.program}", (x.rho_spatial, x.effect_cptac), fontsize=4.5)
        axx.axhline(0, c="k", lw=.4); axx.axvline(0, c="k", lw=.4)
        axx.set_xlabel("IDH-mut spatial ρ (200 µm)"); axx.set_ylabel(f"CPTAC effect ({mod})"); axx.set_title(f"Spatial → population ({mod}); red = replicated")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m8_spatial_vs_cptac.png"), dpi=160); plt.close(fig)
C.log(f"→ {TAB}")
