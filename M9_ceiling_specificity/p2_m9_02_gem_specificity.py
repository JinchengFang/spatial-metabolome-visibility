#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--res", default="200"); ap.add_argument("--n-null", type=int, default=200)
ap.add_argument("--min-genes", type=int, default=3)
ap.add_argument("--n-pc", type=int, default=3, help="first n principal components of each gene set as predictors (few predictors avoid overfitting under spatial CV)")
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--max-rxn", type=int, default=400)
ap.add_argument("--power-feat", type=int, default=15); ap.add_argument("--power-null", type=int, default=100)
ap.add_argument("--max-feat", type=int, default=40, help="maximum metabolites tested per sample (one ion per metabolite, highest Moran's I; then top N by Moran's I)")
ap.add_argument("--with-transport", action="store_true", help="include transporters in the gene sets (default: enzymatic reactions only)")
ap.add_argument("--max-set", type=int, default=150, help="metabolites with more expressed genes than this (near-currency metabolites such as AMP) are not tested")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step7")
RES = [int(r) for r in a.res.split(",")]; t0 = time.time(); plt.rcParams.update({"font.size": 7})
GEM = C.load_human_gem(P["REFDB"])
GEM_GENES = set().union(*GEM["met_genes"].values())
C.log(f"Human-GEM: {len(GEM['table'])} metabolites (compartments merged), {len(GEM_GENES)} metabolic genes")
files = sorted(f for f in glob.glob(os.path.join(P["PROC"], "m9_bins", "*.npz")) if any(f.endswith(f"_{r}um.npz") for r in RES))
if not files:
    raise SystemExit("02_processed/m9_bins/*.npz missing; run M9-01 first")


ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])


def batch_X(Gz, sets):
    """N gene sets of equal size -> (N, n, k) predictor array: first --n-pc principal-component scores of each set (standardized; the genes themselves
    if a set has <= n_pc genes). Exact batched SVD for sets of <= 60 genes, batched randomized SVD otherwise (Halko et al. 2011; 2 power iterations)."""
    out = []; k = a.n_pc
    for c0 in range(0, len(sets), 50):
        A = np.stack([Gz[:, s_] for s_ in sets[c0:c0 + 50]])                  # (N, n, m)
        if A.shape[2] > k:
            if A.shape[2] <= 60:
                U, S_, _ = np.linalg.svd(A, full_matrices=False)
            else:
                Om = np.random.default_rng(1).standard_normal((A.shape[2], k + 8))
                Q = np.linalg.qr(np.matmul(A, Om))[0]
                for _ in range(2):
                    Q = np.linalg.qr(np.matmul(A, np.matmul(A.transpose(0, 2, 1), Q)))[0]
                Ub, S_, _ = np.linalg.svd(np.matmul(Q.transpose(0, 2, 1), A), full_matrices=False); U = np.matmul(Q, Ub)
            A = U[:, :, :k] * S_[:, None, :k]
        out.append(A)
    return np.concatenate(out)


def batch_cv_r2(Xb, y, folds):
    """Batched spatial-block cross-validated ridge regression: Xb (N, n, p), y (n,) -> out-of-fold R² of every set.
    alpha is chosen per set from ALPHAS by leave-one-out (hat matrix) within the training set, as in cv_ridge_gcv; own and random sets are treated identically."""
    N, n, p = Xb.shape; pred = np.empty((N, n)); I_ = np.eye(p)
    for f_ in np.unique(folds):
        te = folds == f_; tr = ~te
        Xt = Xb[:, tr]; mu = Xt.mean(1, keepdims=True); sd = Xt.std(1, keepdims=True) + 1e-9
        Xt = (Xt - mu) / sd; Xs = (Xb[:, te] - mu) / sd
        yt = y[tr] - y[tr].mean()
        XtX = np.matmul(Xt.transpose(0, 2, 1), Xt); Xty = np.matmul(Xt.transpose(0, 2, 1), yt[None, :, None])[:, :, 0]
        best = np.full(N, np.inf); bb = np.zeros((N, p))
        for al in ALPHAS:
            Minv = np.linalg.inv(XtX + al * I_); b = np.matmul(Minv, Xty[:, :, None])[:, :, 0]
            fit = np.matmul(Xt, b[:, :, None])[:, :, 0]; h = (np.matmul(Xt, Minv) * Xt).sum(2)
            loo = (((yt[None, :] - fit) / (1 - h)) ** 2).mean(1)
            m = loo < best; best[m] = loo[m]; bb[m] = b[m]
        pred[:, te] = np.matmul(Xs, bb[:, :, None])[:, :, 0] + y[tr].mean()
    return 1 - ((y[None, :] - pred) ** 2).sum(1) / ((y - y.mean()) ** 2).sum()


def matched_sets(own_idx, uni_idx, dec, n, rng):
    """Random gene sets of the size of own with the same mean-expression decile distribution (drawn from uni, own excluded)."""
    uni = np.setdiff1d(uni_idx, own_idx); by = {d: uni[dec[uni] == d] for d in np.unique(dec[uni])}
    need = pd.Series(dec[own_idx]).value_counts()
    out = []
    for _ in range(n):
        s = []
        for d, k in need.items():
            pool = by.get(d, np.array([], int))
            if len(pool) < k:
                pool = uni
            s.append(rng.choice(pool, k, replace=False))
        out.append(np.concatenate(s))
    return out


rows, prow = [], []
for f in files:
    Z = np.load(f, allow_pickle=True)
    ds, smp, R = str(Z["dataset"]), str(Z["sample"]), int(Z["res"])
    xy = Z["xy"]; Yall = Z["Yf"].astype(float); ann = Z["annotation"].astype(str); q = Z["moran_q"]; I = Z["moran_I"]
    genes = Z["genes"].astype(str); LG = Z["LG"].astype(float); gidx = {g: i for i, g in enumerate(genes)}
    cand = []
    for j in np.where((q < 0.05) & (np.char.str_len(ann) > 0))[0]:
        gs, hit = C.gem_genes_for_annotation(ann[j], GEM, max_rxn=a.max_rxn, with_transport=a.with_transport)
        idx = np.array(sorted(gidx[g] for g in gs if g in gidx), int)
        if a.min_genes <= len(idx) <= a.max_set and hit:
            cand.append((j, idx, "|".join(sorted(hit)), len(gs)))
    # several adducts / isotopes of one metabolite: keep the one with the highest Moran's I
    best = {}
    for j, idx, key, ng in cand:
        if key not in best or I[j] > I[best[key][0]]:
            best[key] = (j, idx, key, ng)
    cand = sorted(best.values(), key=lambda t: -I[t[0]])[: a.max_feat]
    if not cand:
        C.log(f"  {ds} {smp} @ {R} µm: no structured metabolite feature maps to Human-GEM ({int((np.char.str_len(ann) > 0).sum())} annotated)"); continue
    folds = C.spatial_folds(xy, a.block_um, seed=R); T = C.poly2(xy)
    Zc = np.column_stack([np.ones(len(xy)), Z["comp"].astype(float), Z["tech"].astype(float)])
    mean_expr = LG.mean(0); dec = np.minimum((pd.Series(mean_expr).rank(pct=True).values * 10).astype(int), 9)
    U_all = np.arange(len(genes)); U_gem = np.array([gidx[g] for g in genes if g in GEM_GENES], int)
    C.log(f"  {ds} {smp} @ {R} µm: {len(cand)} mappable structured metabolites (median {np.median([len(c[1]) for c in cand]):.0f} expressed genes per set); "
          f"U_all {len(U_all)}, U_gem {len(U_gem)} genes")
    rng = np.random.default_rng(R)
    for level in ("full", "detrended", "beyond_comp"):
        if level == "full":
            Yl, Gl = C.zcols(Yall), C.zcols(LG)
        elif level == "detrended":
            Yl, Gl = C.zcols(C.detrend(Yall, T)), C.zcols(C.detrend(LG, T))
        else:
            Yl, Gl = C.zcols(C.detrend(Yall, Zc)), C.zcols(C.detrend(LG, Zc))
        Gl = np.nan_to_num(Gl)
        for ci, (j, idx, key, ng) in enumerate(cand):
            y = Yl[:, j]
            res = dict(dataset=ds, sample=smp, res_um=R, level=level, feature=str(Z["names"][j]), annotation=ann[j], gem_match=key,
                       n_genes_gem=ng, n_genes_expr=len(idx), genes=";".join(genes[idx][:30]), moran_I=float(I[j]))
            for uname, U in (("all", U_all), ("gem", U_gem)):
                sets = [idx] + matched_sets(idx, U, dec, a.n_null, rng)
                r_ = batch_cv_r2(batch_X(Gl, sets), y, folds); r_own, nul = r_[0], r_[1:]
                res.update({"R2_own": float(r_own), f"null_med_{uname}": float(np.median(nul)), f"null_p95_{uname}": float(np.percentile(nul, 95)),
                            f"pctl_{uname}": float(np.mean(nul < r_own)), f"p_{uname}": float((np.sum(nul >= r_own) + 1) / (len(nul) + 1))})
            rows.append(res)
            # power (positive control): full level, first --power-feat features; own-pathway score added (first PC / mean)
            if level == "full" and ci < a.power_feat:
                score = C.zcols(batch_X(Gl, [idx])[0][:, :1])[:, 0]          # main expression axis of the own pathway
                sets = [idx] + matched_sets(idx, U_gem, dec, a.power_null, rng); Xb = batch_X(Gl, sets)
                for w in (0.05, 0.10, 0.20):
                    yi = np.sqrt(1 - w) * y + np.sqrt(w) * score
                    r_ = batch_cv_r2(Xb, yi, folds)
                    prow.append(dict(dataset=ds, sample=smp, res_um=R, feature=res["feature"], w=w, R2_own=float(r_[0]),
                                     p_gem=float((np.sum(r_[1:] >= r_[0]) + 1) / len(r_))))
        d_ = pd.DataFrame([r for r in rows if r["sample"] == smp and r["res_um"] == R and r["level"] == level])
        C.log(f"     [{level}] median R²_own {d_.R2_own.median():+.3f}; median percentile U_all {d_.pctl_all.median():.2f} / U_gem {d_.pctl_gem.median():.2f}; "
              f"fraction p < 0.05 U_all {np.mean(d_.p_all < 0.05):.2f} / U_gem {np.mean(d_.p_gem < 0.05):.2f}")
if not rows:
    raise SystemExit("no testable metabolites")
D = pd.DataFrame(rows)
for u in ("all", "gem"):
    D[f"q_{u}"] = D.groupby(["dataset", "sample", "res_um", "level"])[f"p_{u}"].transform(lambda v: C.bh(v.values))
D.to_csv(os.path.join(TAB, "m9_gem_specificity.tsv"), sep="\t", index=False)
PW = pd.DataFrame(prow)
if len(PW):
    PW.to_csv(os.path.join(TAB, "m9_gem_power.tsv"), sep="\t", index=False)


def summ(g):
    out = dict(n=len(g), R2_own=g.R2_own.median())
    for u in ("all", "gem"):
        k = int((g[f"p_{u}"] < 0.05).sum())
        out.update({f"pctl_med_{u}": g[f"pctl_{u}"].median(), f"frac_p05_{u}": k / len(g),
                    f"binom_p_{u}": binomtest(k, len(g), 0.05, alternative="greater").pvalue,
                    f"wilcoxon_p_{u}": wilcoxon(g[f"pctl_{u}"] - 0.5, alternative="greater").pvalue if len(g) >= 6 else np.nan})
    return pd.Series(out)


SM = D.groupby(["dataset", "sample", "res_um", "level"]).apply(summ).reset_index()
if len(PW):
    pw = PW.groupby(["dataset", "sample", "res_um", "w"]).p_gem.apply(lambda v: float(np.mean(v < 0.05))).unstack("w")
    pw.columns = [f"power_w{c:g}" for c in pw.columns]
    SM = SM.merge(pw.reset_index(), on=["dataset", "sample", "res_um"], how="left")
pwc = [c for c in SM.columns if c.startswith("power_w")]
SM["MDE_w"] = SM[pwc].apply(lambda r: min([float(c[7:]) for c in pwc if r[c] >= 0.8], default=np.nan), axis=1) if pwc else np.nan   # minimal detectable effect (smallest added variance with power >= 0.8)
pw02 = SM.get("power_w0.2", pd.Series(np.nan, index=SM.index))
spec = SM.binom_p_gem < 0.05                                                    # significantly more than 5% of metabolites beat random metabolic genes
weak = ~spec & (SM.pctl_med_gem >= 0.65) & (SM.wilcoxon_p_gem < 0.05)          # shifted upwards overall, no single metabolite significant
SM["D33_no_specificity"] = ~spec & ~weak & (pw02 >= 0.8)
SM["D33_status"] = np.select([spec, weak, pw02 >= 0.8], ["specificity detected", "weak tendency (higher overall, none individually significant)", "criterion met (no specificity, adequately powered)"], "underpowered (no conclusion)")
SM.to_csv(os.path.join(TAB, "m9_gem_summary.tsv"), sep="\t", index=False)
C.log("summary:\n" + SM.round(3).to_string(index=False))

# figure: percentile per sample x level (U_gem); 0.5 = like random metabolic genes
lv = ["full", "detrended", "beyond_comp"]; R0 = RES[0]
d = D[D.res_um == R0]; keys = d[["dataset", "sample"]].drop_duplicates().values.tolist()
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[3, 1]))
for li, l_ in enumerate(lv):
    for ki, (ds, smp) in enumerate(keys):
        v = d[(d.dataset == ds) & (d["sample"] == smp) & (d.level == l_)].pctl_gem.values
        x = ki * 4 + li
        if len(v):
            axs[0].scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(-.25, .25, len(v)), v, s=6, alpha=.7, c=["#3182bd", "#e6550d", "#31a354"][li])
            axs[0].plot([x - .35, x + .35], [np.median(v)] * 2, c="k", lw=1.2)
axs[0].axhline(0.5, c="grey", ls="--", lw=.6); axs[0].axhline(0.95, c="grey", ls=":", lw=.6)
axs[0].set_xticks([ki * 4 + 1 for ki in range(len(keys))]); axs[0].set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], fontsize=6)
axs[0].set_ylabel("percentile of own-pathway R² among\nexpression-matched random metabolic gene sets")
axs[0].set_title(f"{R0} µm · blue = full, orange = detrended, green = beyond composition")
if len(PW):
    pw_ = PW[PW.res_um == R0].groupby(["sample", "w"]).p_gem.apply(lambda v: np.mean(v < 0.05)).unstack("w")
    for i, (smp, r_) in enumerate(pw_.iterrows()):
        axs[1].plot(r_.index.values, r_.values, "o-", ms=3, lw=.8, label=smp)
    axs[1].set_ylim(0, 1.05); axs[1].set_xlabel("injected own-pathway variance (w)"); axs[1].set_ylabel("detection rate (p<0.05)")
    axs[1].legend(fontsize=5, frameon=False); axs[1].set_title("power (positive control)")
fig.suptitle("Does a metabolite's own GEM pathway predict its spatial distribution better than random metabolic genes?", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FIG, "m9_gem_specificity.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
