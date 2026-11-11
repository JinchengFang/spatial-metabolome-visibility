#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
ap.add_argument("--k-axes", type=int, default=3); ap.add_argument("--n-hvg", type=int, default=2000); ap.add_argument("--min-samples", type=int, default=3)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step9"); R = a.res; t0 = time.time(); plt.rcParams.update({"font.size": 7})
D = {}
for f in sorted(glob.glob(os.path.join(P["PROC"], "m9_bins", f"ccRCC_*_{R}um.npz"))):
    Z = np.load(f, allow_pickle=True); D[str(Z["sample"])] = {k: Z[k] for k in Z.files}
smp = sorted(D)
if len(smp) < a.min_samples:
    C.log(f"fewer than {a.min_samples} ccRCC samples (run Step 7 M9-01 first); skipped"); sys.exit(0)
fa = os.path.join(P["P2"], "05_tables", "Step8", "m11_atlas.tsv")
AT = pd.read_csv(fa, sep="\t") if os.path.exists(fa) else None
if AT is not None and len(AT[AT.res_um == R]):
    AT = AT[AT.res_um == R]; feats = AT.feature.astype(str).tolist()
else:
    common = sorted(set.intersection(*[set(D[s]["names"].astype(str)) for s in smp])); feats = common[:60]
idx = {s: pd.Index(D[s]["names"].astype(str)).get_indexer(feats) for s in smp}
genes = sorted(set.intersection(*[set(D[s]["genes"].astype(str)) for s in smp]))
gix = {s: pd.Index(D[s]["genes"].astype(str)).get_indexer(genes) for s in smp}
hvg = np.argsort(-np.mean([D[s]["LG"][:, gix[s]].astype(float).var(0) for s in smp], axis=0))[: a.n_hvg]
C.log(f"{len(smp)} patients, {len(feats)} metabolite features (from the Step 8 atlas), {len(hvg)} highly variable genes")


def profiles(variant):
    out = {}
    for s in smp:
        Y = C.zcols(D[s]["Yf"][:, idx[s]].astype(float)); G = np.nan_to_num(C.zcols(D[s]["LG"][:, gix[s][hvg]].astype(float)))
        if variant == "axis-removed":
            U, S_, _ = np.linalg.svd(Y, full_matrices=False); T = U[:, : a.k_axes]
            Y = C.zcols(Y - T @ (T.T @ Y))
        out[s] = (Y.T @ G) / len(Y)
    return out


rows = []
for variant in ("raw", "axis-removed"):
    Pf = profiles(variant)
    for i_, s1 in enumerate(smp):
        for s2 in smp:
            if s1 == s2:
                continue
            M = np.corrcoef(Pf[s1], Pf[s2])[: len(feats), len(feats):]
            for k, f in enumerate(feats):
                mis = np.delete(M[k], k)
                rows.append(dict(variant=variant, sample_1=s1, sample_2=s2, feature=f, r_matched=M[k, k], auc=float(np.mean(mis < M[k, k])),
                                 r_mismatched_median=float(np.median(mis))))
ID = pd.DataFrame(rows); ID.to_csv(os.path.join(TAB, "m12_coupling_identity.tsv"), sep="\t", index=False)
per = ID.groupby(["variant", "feature"]).agg(auc=("auc", "mean"), r_matched=("r_matched", "median")).reset_index()
SM = []
for v, g in per.groupby("variant"):
    SM.append(dict(variant=v, n_feat=len(g), auc_median=g.auc.median(), frac_auc_ge_0_75=float((g.auc >= .75).mean()), r_matched_median=g.r_matched.median(),
                   wilcoxon_p=wilcoxon(g.auc - .5, alternative="greater").pvalue))
SM = pd.DataFrame(SM); SM["D36r_specific_coupling_reproducible"] = SM.auc_median >= 0.65
SM.to_csv(os.path.join(TAB, "m12_coupling_summary.tsv"), sep="\t", index=False)
C.log("coupling-profile identification:\n" + SM.round(3).to_string(index=False))
if AT is not None:
    ar = per[per.variant == "axis-removed"].set_index("feature").auc; rw = per[per.variant == "raw"].set_index("feature").auc
    A2 = AT.copy(); A2["auc_raw"] = A2.feature.astype(str).map(rw); A2["auc_axis_removed"] = A2.feature.astype(str).map(ar)
    A2["tier1"] = (A2.n_samples_norm_ge_0_3 >= 3) & (A2.R2_LOPO > 0.1); A2["tier2"] = A2.tier1 & (A2.auc_axis_removed >= 0.75)
    A2 = A2.sort_values(["tier2", "tier1", "R2_within_norm"], ascending=False); A2.to_csv(os.path.join(TAB, "m12_atlas_v2.tsv"), sep="\t", index=False)
    C.log(f"atlas v2: tier 1 {int(A2.tier1.sum())}, tier 2 {int(A2.tier2.sum())}\n" +
          A2[["feature", "annotation", "R2_within_norm", "n_samples_norm_ge_0_3", "R2_LOPO", "auc_raw", "auc_axis_removed", "tier1", "tier2", "top_pos_genes"]].head(15).round(3).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
for ax_, v in zip(axs, ("raw", "axis-removed")):
    g = per[per.variant == v]; ax_.hist(g.auc, bins=20, range=(0, 1), color="#3182bd" if v == "raw" else "#de2d26", alpha=.85)
    ax_.axvline(.5, c="k", ls="--", lw=.7); ax_.set_xlabel("identification AUC (own coupling profile vs other metabolites)"); ax_.set_ylabel("metabolite features")
    ax_.set_title(f"{v}: median AUC {g.auc.median():.2f}", fontsize=7)
fig.suptitle(f"Is each metabolite's transcriptional coupling reproducible across {len(smp)} ccRCC patients? ({R} µm)", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FIG, "m12_coupling_identity.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
