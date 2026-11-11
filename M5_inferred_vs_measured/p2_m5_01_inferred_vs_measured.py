#!/usr/bin/env python3

import argparse, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore", category=RuntimeWarning)

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--n-null", type=int, default=1000)
ap.add_argument("--min-detect", type=int, default=20); ap.add_argument("--tumor-frac", type=float, default=0.5)
ap.add_argument("--block-um", type=float, default=1000.0)
ap.add_argument("--step", default="Step3"); ap.add_argument("--ion-set", default="all", choices=["all", "strict", "protonated"])
ap.add_argument("--skip-population", action="store_true", help="spatial part only (the population part does not depend on the ion set)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); OUT = os.path.join(P["PROC"], "cptac")
GTAG = "" if a.ion_set == "all" else a.ion_set + "_"
plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(0); summary = {}
ALPHAS = np.logspace(-2, 4, 13)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def cv_pred(X, y, folds):
    """Cross-validated out-of-fold predictions for the given fold labels."""
    yhat = np.full(len(y), np.nan)
    for f_ in np.unique(folds):
        te = folds == f_; tr = ~te
        if tr.sum() < 10 or te.sum() < 2:
            continue
        yhat[te] = ridge().fit(X[tr], y[tr]).predict(X[te])
    return yhat


def r2(y, yhat):
    ok = np.isfinite(yhat)
    if ok.sum() < 10:
        return np.nan
    return float(1 - np.sum((y[ok] - yhat[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))


def zcols(D):
    return (D - D.mean()) / (D.std() + 1e-9)


# ============================================================= P1 population (CPTAC)
pop_rows, cv_rows = [], []
if os.path.exists(os.path.join(OUT, "metabolome_lcimputed.parquet")) and not a.skip_population:
    M = pd.read_parquet(os.path.join(OUT, "metabolome_lcimputed.parquet")); Mdet = pd.read_parquet(os.path.join(OUT, "metabolome.parquet")).notna()
    RNA = pd.read_parquet(os.path.join(OUT, "rna.parquet")); PROT = pd.read_parquet(os.path.join(OUT, "protein.parquet"))
    PR = pd.read_csv(os.path.join(OUT, "programs_rna.tsv"), sep="\t", index_col=0)
    ann = pd.read_csv(os.path.join(OUT, "annotations.tsv"), sep="\t", index_col=0)
    ix = M.index.intersection(RNA.index).intersection(PROT.index).intersection(PR.index)
    RNAx = RNA.loc[ix]; RNAx = RNAx.loc[:, (RNAx > 1).mean() >= 0.5]          # expressed in at least half of the tumours (log2 FPKM-UQ+1 > 1)
    RNAx = RNAx.fillna(RNAx.mean())
    PROTx = PROT.loc[ix]; PROTx = PROTx.loc[:, PROTx.notna().mean() >= 0.8]; PROTx = PROTx.fillna(PROTx.mean())
    ZR, ZP = zcols(RNAx), zcols(PROTx)
    covZ = np.column_stack([PR.loc[ix, ["Neuron_cortex", "WM_oligodendrocyte"]].values, ann.reindex(ix).xcell_microenvironment_score.values])
    C.log(f"P1 population: {len(ix)} tumours with metabolome + RNA + protein; RNA {ZR.shape[1]} genes, protein {ZP.shape[1]}")
    folds_rep = [rng.permutation(np.arange(len(ix)) % 5) for _ in range(10)]
    pcaR = PCA(20, random_state=0).fit_transform(ZR.values)   # unsupervised (metabolites not used), fitted on all samples
    for met, enz in C.METAB_ENZYMES.items():
        if met not in M or Mdet.loc[ix, met].sum() < a.min_detect:
            continue
        y = M.loc[ix, met].values
        up, dn = enz["up"].split(), enz["down"].split()
        row = dict(metabolite=met, n=len(ix), n_detected=int(Mdet.loc[ix, met].sum()), up=" ".join(up), down=" ".join(dn))
        for mod, Z in (("rna", ZR), ("protein", ZP)):
            u_, d_ = [g for g in up if g in Z], [g for g in dn if g in Z]
            row[f"genes_{mod}"] = len(u_) + len(d_)
            if not u_ and not d_:
                continue
            sc = (Z[u_].mean(axis=1) if u_ else 0) - (Z[d_].mean(axis=1) if d_ else 0)
            rho = spearmanr(sc, y)[0]; rho_adj, p_adj, _ = C.partial_spearman(y, np.asarray(sc, float), covZ)
            # null of random gene sets (same numbers of up / down genes)
            allg = Z.columns.values; nul = np.empty(a.n_null)
            for k in range(a.n_null):
                ru = rng.choice(allg, len(u_), replace=False) if u_ else []; rd = rng.choice(allg, len(d_), replace=False) if d_ else []
                s0 = (Z[ru].mean(axis=1) if len(ru) else 0) - (Z[rd].mean(axis=1) if len(rd) else 0)
                nul[k] = spearmanr(s0, y)[0]
            row.update({f"rho_{mod}": rho, f"p_{mod}_vs_random": (np.sum(nul >= rho) + 1) / (a.n_null + 1), f"null_mean_{mod}": nul.mean(),
                        f"null_sd_{mod}": nul.std(), f"rho_{mod}_adj": rho_adj, f"p_{mod}_adj": p_adj})
            # cross-validation with all enzymes of the metabolite (RNA or protein) as features
            Xe = Z[u_ + d_].values
            r2s = [r2(y, cv_pred(Xe, y, f)) for f in folds_rep]
            cv_rows.append(dict(metabolite=met, predictor=f"enzymes_{mod}", n_features=Xe.shape[1], cv_R2=np.nanmean(r2s)))
        for name, Xp in (("composition_programs", zcols(PR.loc[ix]).values), ("transcriptome_PC20", pcaR)):
            r2s = [r2(y, cv_pred(Xp, y, f)) for f in folds_rep]
            cv_rows.append(dict(metabolite=met, predictor=name, n_features=Xp.shape[1], cv_R2=np.nanmean(r2s)))
        pop_rows.append(row)
    PO = pd.DataFrame(pop_rows); CVp = pd.DataFrame(cv_rows)
    if len(PO):
        for mod in ("rna", "protein"):
            if f"p_{mod}_vs_random" in PO:
                PO[f"q_{mod}_vs_random"] = C.bh(PO[f"p_{mod}_vs_random"].fillna(1).values)
        PO.to_csv(os.path.join(TAB, "m5_population_enzyme_scores.tsv"), sep="\t", index=False)
        CVp.to_csv(os.path.join(TAB, "m5_population_cv.tsv"), sep="\t", index=False)
        both = PO.dropna(subset=["rho_rna", "rho_protein"])
        w = wilcoxon(both.rho_protein, both.rho_rna) if len(both) >= 5 else None
        summary["population"] = dict(n_metabolites=len(PO), n_tumors=len(ix),
                                     rna_beats_random_q05=int((PO.get("q_rna_vs_random", pd.Series(dtype=float)) < 0.05).sum()),
                                     protein_beats_random_q05=int((PO.get("q_protein_vs_random", pd.Series(dtype=float)) < 0.05).sum()),
                                     median_rho_rna=float(PO.rho_rna.median()), median_rho_protein=float(PO.get("rho_protein", pd.Series(dtype=float)).median()),
                                     protein_vs_rna_wilcoxon_p=float(w.pvalue) if w else None,
                                     median_cv_R2=CVp.groupby("predictor").cv_R2.median().round(3).to_dict())
        C.log(f"  {len(PO)} metabolites: enzyme score better than random gene sets (q < 0.05) RNA {summary['population']['rna_beats_random_q05']}, protein {summary['population']['protein_beats_random_q05']}; "
              f"median rho RNA {PO.rho_rna.median():+.2f} / protein {PO.get('rho_protein', pd.Series(dtype=float)).median():+.2f}"
              + (f"; protein vs RNA Wilcoxon p={w.pvalue:.2g}" if w else ""))
        C.log("  median cross-validated R² by predictor: " + ", ".join(f"{k} {v:+.3f}" for k, v in summary["population"]["median_cv_R2"].items()))
        for r in PO.sort_values("rho_rna", ascending=False).itertuples():
            C.log(f"     {r.metabolite:28s} RNA rho={r.rho_rna:+.2f} (random {r.null_mean_rna:+.2f}+-{r.null_sd_rna:.2f}, p={r.p_rna_vs_random:.2g})"
                  + (f"  protein rho={r.rho_protein:+.2f} (p={r.p_protein_vs_random:.2g})" if hasattr(r, "rho_protein") and np.isfinite(getattr(r, "rho_protein", np.nan)) else "  protein -"))
        fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2))
        if "rho_protein" in PO:
            axs[0].scatter(PO.rho_rna, PO.rho_protein, s=14)
            for r in PO.itertuples():
                axs[0].annotate(r.metabolite.replace("acid", "").strip()[:18], (r.rho_rna, r.rho_protein), fontsize=4.5)
            lim = [-.6, .6]; axs[0].plot(lim, lim, "k--", lw=.5); axs[0].axhline(0, c="k", lw=.3); axs[0].axvline(0, c="k", lw=.3)
            axs[0].set_xlabel("enzyme score ρ (RNA)"); axs[0].set_ylabel("enzyme score ρ (protein)"); axs[0].set_title("Producer − consumer score vs measured metabolite")
        piv = CVp.pivot(index="metabolite", columns="predictor", values="cv_R2")
        axs[1].boxplot([piv[c].dropna() for c in piv.columns]); axs[1].set_xticks(range(1, piv.shape[1] + 1)); axs[1].set_xticklabels(piv.columns, rotation=30, ha="right")
        axs[1].axhline(0, c="k", lw=.4); axs[1].set_ylabel("cross-validated R²"); axs[1].set_title("CPTAC: what predicts measured metabolites?")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "m5_population_rna_vs_protein.png"), dpi=160); plt.close(fig)
else:
    C.log("population part (P1) skipped (--skip-population)" if a.skip_population else "no CPTAC data (run M8-01 first); P1 skipped")

# ============================================================= P2 spatial (IDH-mutant, bin matrices of M4-01)
sp_rows = []
imap_f = os.path.join(TAB, "m4_ion_map.tsv")
import re
grids = sorted([f for f in os.listdir(P["PROC"]) if re.fullmatch(rf"m4_grid_IDHmut_{GTAG}\d+um\.parquet", f)], key=lambda s: int(s.split("_")[-1][:-10]))
if grids and os.path.exists(imap_f):
    IM = pd.read_csv(imap_f, sep="\t")
    for f in grids:
        R = int(f.split("_")[-1][:-10]); G = pd.read_parquet(os.path.join(P["PROC"], f))
        mcols = [c for c in G if c.startswith("m:")]; ctc = [c for c in G if c.startswith("ct:")]; gcols = [c for c in G if c.startswith("g:")]
        gcols = [c for c in gcols if G[c].std() > 0]
        blk = max(3, int(round(a.block_um / R)))
        bid = (G.gy.values // blk) * 10000 + (G.gx.values // blk); ub = np.unique(bid)
        fold_of_block = dict(zip(ub, np.random.default_rng(R).permutation(np.arange(len(ub)) % 5)))
        folds = np.array([fold_of_block[b] for b in bid])
        tumor = (G["lin:Malignant"].values >= a.tumor_frac) if "lin:Malignant" in G else np.zeros(len(G), bool)
        Xc = G[ctc].values; Xg = G[gcols].values; Xcg = np.column_stack([Xc, Xg])
        for subset, msk in (("all", np.ones(len(G), bool)), ("tumor", tumor)):
            if msk.sum() < 80 or len(np.unique(folds[msk])) < 3:
                continue
            for mc in mcols:
                met = mc[2:]; y = G.loc[msk, mc].values; fo = folds[msk]
                res = dict(res_um=R, subset=subset, metabolite=met, n_bins=int(msk.sum()))
                res["R2_composition"] = r2(y, cv_pred(Xc[msk], y, fo))
                res["R2_genes"] = r2(y, cv_pred(Xg[msk], y, fo))
                res["R2_comp_genes"] = r2(y, cv_pred(Xcg[msk], y, fo))
                own = ["ion:" + i_ for i_ in IM.ion[IM.metabolite == met] if "ion:" + i_ in G]
                other = [c for c in G if c.startswith("ion:") and c not in own]
                res["R2_other_metabolites"] = r2(y, cv_pred(G.loc[msk, other].values, y, fo)) if other else np.nan
                if len(own) >= 2:   # technical reference: predict one adduct from the other adducts (leave one out, mean)
                    vals = []
                    for o_ in own:
                        yy = G.loc[msk, o_].values; xx = G.loc[msk, [c for c in own if c != o_]].values
                        vals.append(r2(yy, cv_pred(xx, yy, fo)))
                    res["R2_adduct_ceiling"] = float(np.nanmean(vals))
                else:
                    res["R2_adduct_ceiling"] = np.nan
                sp_rows.append(res)
        C.log(f"P2 spatial [{R} µm]: {len(G)} bins (within tumour {int(tumor.sum())}), {len(mcols)} metabolites, {len(ctc)} cell types + {len(gcols)} genes; block CV {blk}x{blk} bins")
    SPv = pd.DataFrame(sp_rows)
    if len(SPv):
        # internal MALDI reference = max(adduct cross-prediction, prediction from other metabolites); adduct cross-prediction depends on Na/K distribution and is not a strict ceiling
        ref = SPv[["R2_adduct_ceiling", "R2_other_metabolites"]].max(axis=1)
        SPv["maldi_internal_reference"] = ref
        SPv["frac_of_maldi_reference"] = SPv.R2_comp_genes.clip(lower=0) / ref.where(ref > 0.05)
        SPv["genes_beyond_composition"] = SPv.R2_comp_genes - SPv.R2_composition
        SPv.to_csv(os.path.join(TAB, "m5_spatial_cv.tsv"), sep="\t", index=False)
        agg = SPv.groupby(["res_um", "subset"])[["R2_composition", "R2_genes", "R2_comp_genes", "genes_beyond_composition", "R2_other_metabolites",
                                                 "R2_adduct_ceiling", "maldi_internal_reference", "frac_of_maldi_reference"]].median().round(3)
        C.log("spatial-block cross-validated R² (median across metabolites):\n" + agg.to_string())
        summary["spatial"] = {f"{k[0]}um_{k[1]}": v for k, v in agg.to_dict(orient="index").items()}
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
        for axx, sub in zip(axs, ("all", "tumor")):
            h = SPv[SPv.subset == sub]
            if not len(h):
                continue
            cols = ["R2_composition", "R2_genes", "R2_comp_genes", "R2_other_metabolites", "R2_adduct_ceiling"]
            for k_, R in enumerate(sorted(h.res_um.unique())):
                med = h[h.res_um == R][cols].median()
                axx.bar(np.arange(len(cols)) + k_ * 0.27, med.values, width=0.27, label=f"{R} µm")
            axx.set_xticks(np.arange(len(cols)) + .27); axx.set_xticklabels(["composition", "panel genes", "comp+genes", "other metabolites", "adduct cross-pred."], rotation=30, ha="right")
            axx.axhline(0, c="k", lw=.4); axx.set_ylabel("spatial block-CV R² (median)"); axx.set_title(f"IDH-mut: predicting measured metabolites ({sub})"); axx.legend(fontsize=6, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "m5_spatial_explained_variance.png"), dpi=160); plt.close(fig)
else:
    C.log("no bin matrices of M4-01 (02_processed/m4_grid_IDHmut_*um.parquet); P2 skipped")
C.save_json(summary, os.path.join(TAB, "m5_summary.json"))
C.log(f"→ {TAB}")
