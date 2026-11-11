#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd
from math import ceil
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, binom, poisson
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--res", default="200,400"); ap.add_argument("--max-feat", type=int, default=60)
ap.add_argument("--n-hvg", type=int, default=2000); ap.add_argument("--n-pc", type=int, default=30); ap.add_argument("--block-um", type=float, default=1000.0)
ap.add_argument("--cluster-r", type=float, default=0.9); ap.add_argument("--min-samples", type=int, default=3)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step8"); cfg = L.Config(P)
RES = [int(r) for r in a.res.split(",")]; t0 = time.time(); plt.rcParams.update({"font.size": 7})
HALL = cfg.HALL or cfg.STATE_FALLBACK


def load(R):
    out = {}
    for f in sorted(glob.glob(os.path.join(P["PROC"], "m9_bins", f"ccRCC_*_{R}um.npz"))):
        Z = np.load(f, allow_pickle=True); out[str(Z["sample"])] = {k: Z[k] for k in Z.files}
    return out


T_rows, S_rows, REP, ATL = [], [], [], []
for R in RES:
    D = load(R); smp = sorted(D)
    if len(smp) < a.min_samples:
        C.log(f"[{R} µm] fewer than {a.min_samples} ccRCC samples ({smp}); skipped (run Step 7 M9-01 first)"); continue
    n = len(smp); min_struct = max(2, ceil(0.6 * n))
    common = set.intersection(*[set(D[s]["names"].astype(str)) for s in smp])
    idx = {s: {nm: i for i, nm in enumerate(D[s]["names"].astype(str))} for s in smp}
    cand = [nm for nm in sorted(common) if sum(D[s]["moran_q"][idx[s][nm]] < 0.05 for s in smp) >= min_struct]
    Zs = {s: C.zcols(D[s]["Yf"][:, [idx[s][nm] for nm in cand]].astype(float)) for s in smp}
    pooled = np.vstack([Zs[s] for s in smp]); mI = np.array([np.median([D[s]["moran_I"][idx[s][nm]] for s in smp]) for nm in cand])
    Cr = np.nan_to_num(np.corrcoef(pooled.T))
    lab = fcluster(linkage(squareform(1 - np.abs(Cr), checks=False), "average"), t=1 - a.cluster_r, criterion="distance")
    reps = sorted([np.where(lab == k)[0][np.argmax(mI[lab == k])] for k in np.unique(lab)], key=lambda j: -mI[j])[: a.max_feat]
    feats = [cand[j] for j in reps]
    genes = sorted(set.intersection(*[set(D[s]["genes"].astype(str)) for s in smp]))
    gix = {s: pd.Index(D[s]["genes"].astype(str)).get_indexer(genes) for s in smp}
    var_mean = np.mean([D[s]["LG"][:, gix[s]].astype(float).var(0) for s in smp], axis=0)
    hvg = np.argsort(-var_mean)[: a.n_hvg]
    C.log(f"[{R} µm] {n} patients {smp}; {len(common)} shared metabolite features -> {len(cand)} structured in >= {min_struct} patients -> {len(feats)} after merging; {len(genes)} shared genes, {len(hvg)} highly variable")
    ann = {nm: next((str(D[s]["annotation"][idx[s][nm]]) for s in smp if str(D[s]["annotation"][idx[s][nm]])), "") for nm in feats}
    for level in ("full", "detrended"):
        Y, G, H, folds, rel = {}, {}, {}, {}, {}
        for s in smp:
            T = C.poly2(D[s]["xy"]); prep = (lambda M_: C.zcols(M_)) if level == "full" else (lambda M_: C.zcols(C.detrend(M_, T)))
            j = [idx[s][nm] for nm in feats]
            Y[s] = prep(D[s]["Yf"][:, j].astype(float))
            G[s] = np.nan_to_num(prep(D[s]["LG"][:, gix[s][hvg]].astype(float)))
            sc, _ = C.score_programs(D[s]["LG"].astype(float), pd.Index(D[s]["genes"].astype(str)), HALL, min_genes=10 if cfg.HALL else 3)
            H[s] = sc
            folds[s] = C.spatial_folds(D[s]["xy"], a.block_um, seed=R)
            rel[s] = C.split_half_reliability(prep(D[s]["YA"][:, j].astype(float)), prep(D[s]["YB"][:, j].astype(float)))[1]
        hcols = sorted(set.intersection(*[set(H[s].columns) for s in smp]))
        for s in smp:
            T = C.poly2(D[s]["xy"]); Hv = H[s][hcols].values
            H[s] = np.nan_to_num(C.zcols(Hv) if level == "full" else C.zcols(C.detrend(Hv, T)))
        for pred in ("HVG_PC", "Hallmark"):
            for h in smp:
                tr = [s for s in smp if s != h]
                if pred == "HVG_PC":
                    pca = PCA(a.n_pc, random_state=0).fit(np.vstack([G[s] for s in tr]))
                    Xw = PCA(a.n_pc, random_state=0).fit_transform(G[h]); Xtr = np.vstack([pca.transform(G[s]) for s in tr]); Xte = pca.transform(G[h])
                else:
                    Xw = H[h]; Xtr = np.vstack([H[s] for s in tr]); Xte = H[h]
                r_w = C.r2_columns(Y[h], C.cv_ridge_gcv(Xw, Y[h], folds[h]))
                Xall = np.vstack([Xtr, Xte]); Yall = np.vstack([np.vstack([Y[s] for s in tr]), Y[h]])
                fo = np.r_[np.zeros(len(Xtr), int), np.ones(len(Xte), int)]
                Yh = C.cv_ridge_gcv(Xall, Yall, fo)[fo == 1]
                r_l = C.r2_columns(Y[h], Yh); c_l = np.array([np.corrcoef(Y[h][:, k], Yh[:, k])[0, 1] for k in range(len(feats))])
                for k, nm in enumerate(feats):
                    T_rows.append(dict(res_um=R, level=level, predictor=pred, sample=h, feature=nm, annotation=ann[nm], reliability_SB=rel[h][k],
                                       R2_within=r_w[k], R2_LOPO=r_l[k], r_LOPO=c_l[k], R2_within_norm=r_w[k] / rel[h][k] if rel[h][k] > 0.1 else np.nan))
        # reproducibility of coupling profiles (HVG correlation vectors)
        prof = {s: (Y[s].T @ G[s]) / len(Y[s]) for s in smp}               # feature x gene
        matched, mism = [], []
        for i_, s1 in enumerate(smp):
            for s2 in smp[i_ + 1:]:
                M_ = np.corrcoef(prof[s1], prof[s2])[: len(feats), len(feats):]
                matched.append(np.diag(M_)); mism.append(M_[~np.eye(len(feats), dtype=bool)])
        matched = np.median(np.array(matched), 0); mism = np.concatenate(mism); thr = np.percentile(mism, 95)
        rk = []
        d_ = pd.DataFrame([r for r in T_rows if r["res_um"] == R and r["level"] == level and r["predictor"] == "HVG_PC"])
        W = d_.pivot(index="feature", columns="sample", values="R2_within")
        for i_, s1 in enumerate(smp):
            for s2 in smp[i_ + 1:]:
                rk.append(spearmanr(W[s1], W[s2]).correlation)
        for k, nm in enumerate(feats):
            REP.append(dict(res_um=R, level=level, feature=nm, annotation=ann[nm], profile_r_matched=matched[k], null_p95=thr,
                            reproducible=bool(matched[k] > thr)))
        for pred in ("HVG_PC", "Hallmark"):
            e = pd.DataFrame([r for r in T_rows if r["res_um"] == R and r["level"] == level and r["predictor"] == pred])
            row = dict(res_um=R, level=level, predictor=pred, n_samples=n, n_feat=len(feats), R2_within=e.R2_within.median(), R2_LOPO=e.R2_LOPO.median(),
                       r_LOPO=e.r_LOPO.median(), frac_LOPO_pos=float((e.R2_LOPO > 0).mean()), R2_within_norm=e.R2_within_norm.median(),
                       transport_ratio=e.R2_LOPO.median() / e.R2_within.median() if e.R2_within.median() > 0.01 else np.nan)
            if pred == "HVG_PC":
                row.update(profile_r_matched=float(np.median(matched)), profile_r_mismatch_p95=float(thr), frac_reproducible=float(np.mean(matched > thr)),
                           predictability_rank_rho=float(np.nanmedian(rk)))
                row["D36_coupling_reproducible"] = row["frac_reproducible"] >= 0.5
            row["D35_not_transportable"] = bool(row["R2_LOPO"] < 0.5 * row["R2_within"]) if row["R2_within"] > 0.01 else np.nan
            S_rows.append(row)
            C.log(f"   [{level}, {pred}] median R²_within {row['R2_within']:+.3f} (/ceiling {row['R2_within_norm']:.2f}); R²_LOPO {row['R2_LOPO']:+.3f}; r_LOPO {row['r_LOPO']:+.2f}"
                  + (f"; matched profile r {row['profile_r_matched']:.2f} vs unmatched 95% {thr:.2f} -> reproducible {row['frac_reproducible']:.0%}; predictability rank correlation {row['predictability_rank_rho']:.2f}" if pred == "HVG_PC" else ""))
        # atlas (full level): top genes pooled across patients (mean Fisher z) and Hallmark sets
        if level == "full":
            gn = np.array(genes)[hvg]
            zg = np.mean([np.arctanh(np.clip(prof[s], -.999, .999)) for s in smp], 0)
            hp = np.mean([np.arctanh(np.clip((Y[s].T @ H[s]) / len(Y[s]), -.999, .999)) for s in smp], 0)
            e = pd.DataFrame([r for r in T_rows if r["res_um"] == R and r["level"] == level and r["predictor"] == "HVG_PC"])
            for k, nm in enumerate(feats):
                f_ = e[e.feature == nm]
                ATL.append(dict(res_um=R, feature=nm, annotation=ann[nm], n_samples=n, reliability_SB=f_.reliability_SB.median(),
                                R2_within=f_.R2_within.median(), R2_within_norm=f_.R2_within_norm.median(),
                                n_samples_norm_ge_0_3=int((f_.R2_within_norm >= 0.3).sum()), R2_LOPO=f_.R2_LOPO.median(), r_LOPO=f_.r_LOPO.median(),
                                profile_r=matched[k], reproducible=bool(matched[k] > thr),
                                top_pos_genes=";".join(gn[np.argsort(-zg[k])[:8]]), top_neg_genes=";".join(gn[np.argsort(zg[k])[:8]]),
                                top_hallmark="; ".join(f"{hcols[t]} ({np.tanh(hp[k, t]):+.2f})" for t in np.argsort(-np.abs(hp[k]))[:3])))
TR, SM, RP, AT = pd.DataFrame(T_rows), pd.DataFrame(S_rows), pd.DataFrame(REP), pd.DataFrame(ATL)
if not len(SM):
    raise SystemExit("no results (m9_bins of >= 3 ccRCC samples needed)")
AT["candidate"] = (AT.n_samples_norm_ge_0_3 >= np.ceil(0.6 * AT.n_samples)) & (AT.R2_LOPO > 0.1) & AT.reproducible
AT = AT.sort_values(["res_um", "candidate", "R2_within_norm"], ascending=[True, False, False])
for df, f in ((TR, "m11_transport.tsv"), (SM, "m11_transport_summary.tsv"), (RP, "m11_coupling_reproducibility.tsv"), (AT, "m11_atlas.tsv")):
    df.to_csv(os.path.join(TAB, f), sep="\t", index=False)
C.log("summary:\n" + SM.round(3).to_string(index=False))
C.log(f"candidates (R²/ceiling >= 0.3 in >= 60% of patients, LOPO R² > 0.1, reproducible coupling profile): {int(AT.candidate.sum())}\n"
      + AT[AT.candidate][["res_um", "feature", "annotation", "R2_within_norm", "R2_LOPO", "profile_r", "top_pos_genes"]].head(20).round(3).to_string(index=False))

# consistency of GEM pathway specificity across patients
fg = os.path.join(P["P2"], "05_tables", "Step7", "m9_gem_specificity.tsv")
if os.path.exists(fg):
    GD = pd.read_csv(fg, sep="\t"); GD = GD[GD.dataset == "ccRCC"]; grows = []
    for (R, lv), g in GD.groupby(["res_um", "level"]):
        per = g.groupby("feature").agg(n=("sample", "nunique"), k=("p_gem", lambda v: int((v < 0.05).sum())), pctl=("pctl_gem", "mean"), ann=("annotation", "first"))
        per = per[per.n >= 2]
        exp = float(sum(binom.sf(1, int(nn), 0.05) for nn in per.n)); obs = int((per.k >= 2).sum())
        piv = g.pivot_table(index="feature", columns="sample", values="pctl_gem"); rr = []
        cols = list(piv.columns)
        for i_, c1 in enumerate(cols):
            for c2 in cols[i_ + 1:]:
                m = piv[[c1, c2]].dropna()
                if len(m) >= 5:
                    rr.append(spearmanr(m[c1], m[c2]).correlation)
        grows.append(dict(res_um=R, level=lv, n_metabolites_in_ge2=len(per), observed_sig_in_ge2=obs, expected_by_chance=exp,
                          poisson_p=float(poisson.sf(obs - 1, exp)) if exp > 0 else np.nan, pctl_pairwise_rho=float(np.nanmedian(rr)) if rr else np.nan,
                          replicated=";".join(per[per.k >= 2].ann.astype(str).str[:40].tolist())))
    GC = pd.DataFrame(grows); GC.to_csv(os.path.join(TAB, "m11_gem_consistency.tsv"), sep="\t", index=False)
    C.log("consistency of GEM pathway specificity across patients:\n" + GC.round(3).to_string(index=False))

# figures
R0 = RES[0]; e = TR[(TR.res_um == R0) & (TR.level == "full") & (TR.predictor == "HVG_PC")]
if len(e):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
    for s, g in e.groupby("sample"):
        axs[0].scatter(g.R2_within, g.R2_LOPO, s=6, alpha=.6, label=s)
    lim = [min(-.2, e.R2_LOPO.min()), max(.8, e.R2_within.max())]
    axs[0].plot(lim, lim, "k--", lw=.6); axs[0].axhline(0, c="grey", lw=.4); axs[0].set_xlabel("R² within patient (spatial CV)")
    axs[0].set_ylabel("R² leave-one-patient-out"); axs[0].legend(fontsize=5, frameon=False); axs[0].set_title("a  is the gene→metabolite map transportable?")
    rp = RP[(RP.res_um == R0) & (RP.level == "full")]
    axs[1].hist(rp.profile_r_matched, bins=20, color="#3182bd", alpha=.8, label="same metabolite, different patients")
    axs[1].axvline(rp.null_p95.iloc[0], c="k", ls="--", lw=.8, label="mismatched pairs, 95th pct")
    axs[1].set_xlabel("correlation of coupling profiles (2000 HVGs)"); axs[1].set_ylabel("metabolite features"); axs[1].legend(fontsize=5.5, frameon=False)
    axs[1].set_title("b  coupling-profile reproducibility")
    at = AT[AT.res_um == R0].head(15).iloc[::-1]
    axs[2].barh(np.arange(len(at)), at.R2_within_norm, color=np.where(at.candidate, "#de2d26", "#bdbdbd"))
    axs[2].scatter(at.R2_LOPO / at.reliability_SB, np.arange(len(at)), marker="|", c="k", s=60, label="LOPO / ceiling")
    axs[2].set_yticks(np.arange(len(at))); axs[2].set_yticklabels([(str(r.annotation)[:38] or f"m/z {float(r.feature):.3f}") for r in at.itertuples()], fontsize=5)
    axs[2].set_xlabel("R² / noise ceiling (median across patients)"); axs[2].legend(fontsize=5.5, frameon=False); axs[2].set_title("c  most transcriptome-visible metabolites (red = candidate)")
    fig.suptitle(f"ccRCC, {e['sample'].nunique()} patients, {R0} µm", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FIG, "m11_transport_atlas.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
