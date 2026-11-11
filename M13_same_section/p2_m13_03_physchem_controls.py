#!/usr/bin/env python3

import argparse, glob, os, re, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200); ap.add_argument("--n-perm", type=int, default=499)
ap.add_argument("--max-feat", type=int, default=200); ap.add_argument("--cluster-r", type=float, default=0.9); ap.add_argument("--min-samples", type=int, default=3)
ap.add_argument("--sma-min-samples", type=int, default=2, help="also test SMA sections of one matrix (shared peak list) if there are at least this many")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step10"); R = a.res; rng = np.random.default_rng(0); t0 = time.time()
plt.rcParams.update({"font.size": 7})
GROUPS = {}
for f in sorted(glob.glob(os.path.join(P["PROC"], "m12_residuals", f"*_{R}um.npz"))):
    Z = np.load(f, allow_pickle=True); ds = str(Z["dataset"]); smp = str(Z["sample"])
    g = "ccRCC" if ds == "ccRCC" else (f"SMA-{smp.rsplit('_', 1)[-1]}" if ds == "SMA-mouse" else None)
    if g:
        GROUPS.setdefault(g, {})[smp] = {k: Z[k] for k in Z.files}
SAME_POL = [1.00336, 2.00671, 21.98194, 37.95588, 15.97394, 18.01056, 17.02655, 35.97668, 43.98983, 46.00548, 27.99491, 33.98772]
OPP_POL = [2.01456, 23.99650, 39.97044, 33.96157, 11.98007, 3.99431, 15.99600]


def lipid_class(ann):
    m = re.match(r"^(FA|CAR|LPC|LPE|LPI|LPG|LPS|PC|PE|PS|PI|PA|PG|SM|Cer|HexCer|ST|CE|TG|DG|MG|SHexCer|NAE)\s", str(ann).split(";")[0].strip())
    return m.group(1) if m else ("small molecule" if str(ann).strip() not in ("", "nan") else "unannotated")


def run_group(gname, st):
    cc = sorted(st)
    common = sorted(set.intersection(*[set(st[s]["names"].astype(str)) for s in cc]))
    ix = {s: pd.Index(st[s]["names"].astype(str)).get_indexer(common) for s in cc}
    nstr = np.sum([st[s]["q_moran_Y"][ix[s]] < 0.05 for s in cc], axis=0)
    cand = np.where(nstr >= max(2, int(np.ceil(0.6 * len(cc)))))[0]
    Zp = np.vstack([st[s]["Z"][:, ix[s][cand]] for s in cc]); mI = np.median([st[s]["moran_Y"][ix[s][cand]] for s in cc], axis=0)
    lab = fcluster(linkage(squareform(1 - np.abs(np.nan_to_num(np.corrcoef(Zp.T))), checks=False), "average"), t=1 - a.cluster_r, criterion="distance")
    reps = sorted([np.where(lab == k)[0][np.argmax(mI[lab == k])] for k in np.unique(lab)], key=lambda j: -mI[j])[: a.max_feat]
    F = cand[reps]; s0 = cc[0]
    mz = st[s0]["mz"][ix[s0][F]].astype(float); pol = st[s0]["pol"][ix[s0][F]].astype(str); cls = np.array([lipid_class(x) for x in st[s0]["ann"][ix[s0][F]]])
    n = len(F); iu = np.triu_indices(n, 1)
    D = np.abs(mz[:, None] - mz[None, :]); tol = 0.003 + 5e-6 * np.maximum(mz[:, None], mz[None, :]); same = pol[:, None] == pol[None, :]
    chem = np.zeros((n, n), bool)
    for d in SAME_POL:
        chem |= same & (np.abs(D - d) < tol)
    for d in OPP_POL:
        chem |= ~same & (np.abs(D - d) < tol)
    md_ = mz - np.round(mz)
    COV = np.column_stack([np.abs(np.log(mz[:, None]) - np.log(mz[None, :]))[iu], np.abs(md_[:, None] - md_[None, :])[iu], same[iu].astype(float),
                           (cls[:, None] == cls[None, :])[iu].astype(float)])
    keep = ~chem[iu]
    SUBSETS = {"all": keep, "cross_polarity": keep & ~same[iu], "cross_class": keep & (cls[:, None] != cls[None, :])[iu],
               "dmz_gt_200": keep & (D[iu] > 200), "dmassdefect_gt_0.15": keep & (COV[:, 1] > 0.15)}
    C.log(f"{len(cc)} patients, {n} metabolite features (positive {int((pol == 'positive').sum())} / negative {int((pol == 'negative').sum())}); "
          + "; ".join(f"{k} {int(v.sum())} pairs" for k, v in SUBSETS.items()))
    Cres = {s: np.nan_to_num(np.corrcoef(st[s]["E"][:, ix[s][F]].T)) for s in cc}
    Craw = {s: np.nan_to_num(np.corrcoef(st[s]["Z"][:, ix[s][F]].T)) for s in cc}
    Rk = np.column_stack([np.ones(len(COV))] + [rankdata(COV[:, j]) for j in range(COV.shape[1])])


    def resid(v, m):
        r = rankdata(v); X_ = Rk[m]; return r - X_ @ np.linalg.lstsq(X_, r, rcond=None)[0]


    def corr(u, v):
        u = u - u.mean(); v = v - v.mean(); return float((u @ v) / np.sqrt((u @ u) * (v @ v) + 1e-12))


    rows = []
    for i_, s1 in enumerate(cc):
        for s2 in cc[i_ + 1:]:
            for kind, Cm in (("residual", Cres), ("raw", Craw)):
                v1 = Cm[s1][iu]; perms = [rng.permutation(n) for _ in range(a.n_perm)]
                for sub, m in list(SUBSETS.items()) + [("partial_mantel", keep)]:
                    if m.sum() < 30:
                        continue
                    if sub == "partial_mantel":
                        obs = corr(resid(v1[m], m), resid(Cm[s2][iu][m], m))
                        null = [corr(resid(v1[m], m), resid(Cm[s2][np.ix_(p_, p_)][iu][m], m)) for p_ in perms]
                    else:
                        obs = corr(rankdata(v1[m]), rankdata(Cm[s2][iu][m]))
                        null = [corr(rankdata(v1[m]), rankdata(Cm[s2][np.ix_(p_, p_)][iu][m])) for p_ in perms]
                    rows.append(dict(sample_1=s1, sample_2=s2, kind=kind, subset=sub, n_pairs=int(m.sum()), rho=obs,
                                     p=(np.sum(np.array(null) >= obs) + 1) / (a.n_perm + 1)))
    PR = pd.DataFrame(rows)
    if not len(PR):
        C.log(f"[{gname}] too few feature pairs (< 30); skipped"); return None, None
    SM = PR.groupby(["kind", "subset"]).agg(n_patient_pairs=("rho", "size"), n_feature_pairs=("n_pairs", "median"), rho_median=("rho", "median"),
                                             frac_p05=("p", lambda v: float(np.mean(v < 0.05)))).reset_index()
    vbar = np.mean([Cres[s][iu] for s in cc], 0)[keep]
    X_ = np.column_stack([np.ones(keep.sum()), COV[keep]]); r2_phys = 1 - ((vbar - X_ @ np.linalg.lstsq(X_, vbar, rcond=None)[0]) ** 2).sum() / ((vbar - vbar.mean()) ** 2).sum()
    SM["physchem_R2_of_mean_residual_cov"] = r2_phys
    res = SM[SM.kind == "residual"].set_index("subset")
    d40 = bool(("cross_polarity" in res.index and res.loc["cross_polarity", "rho_median"] >= 0.2 and res.loc["cross_polarity", "frac_p05"] >= 0.7)
               and res.loc["partial_mantel", "rho_median"] >= 0.2 and res.loc["partial_mantel", "frac_p05"] >= 0.7)
    SM["D40_not_ionization_physchem"] = d40
    C.log(f"[{gname}] summary:\n" + SM.round(3).to_string(index=False) + f"\nR² of mean residual covariation explained by physicochemical similarity = {r2_phys:.3f}; D40 = {d40}")
    return PR, SM


ALLP, ALLS = [], []
for gname, st in GROUPS.items():
    need = a.min_samples if gname == "ccRCC" else a.sma_min_samples
    if len(st) < need:
        C.log(f"[{gname}] {len(st)} samples < {need}; skipped"); continue
    PR, SM = run_group(gname, st)
    if PR is None:
        continue
    PR.insert(0, "group", gname); SM.insert(0, "group", gname); ALLP.append(PR); ALLS.append(SM)
if not ALLS:
    C.log("no testable groups (ccRCC needs residual files of >= 3 patients; run M12-01 first)"); sys.exit(0)
PR = pd.concat(ALLP, ignore_index=True); SM = pd.concat(ALLS, ignore_index=True)
PR.to_csv(os.path.join(TAB, "m13_physchem_pairs.tsv"), sep="\t", index=False); SM.to_csv(os.path.join(TAB, "m13_physchem_summary.tsv"), sep="\t", index=False)
g0 = "ccRCC" if "ccRCC" in set(PR.group) else PR.group.iloc[0]
PR = PR[PR.group == g0]; cc = sorted(set(PR.sample_1) | set(PR.sample_2))
fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))
order = ["all", "partial_mantel", "cross_polarity", "cross_class", "dmz_gt_200", "dmassdefect_gt_0.15"]
for k_, (kind, col) in enumerate((("raw", "#9ecae1"), ("residual", "#de2d26"))):
    for j, sub in enumerate(order):
        v = PR[(PR.kind == kind) & (PR.subset == sub)].rho.values
        if len(v):
            ax.scatter(np.full(len(v), j + (k_ - .5) * .3), v, s=10, c=col, label=kind if j == 0 else None)
ax.axhline(0, c="k", lw=.5); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, fontsize=6)
ax.set_ylabel("cross-patient conservation (Mantel ρ)"); ax.legend(fontsize=6, frameon=False)
ax.set_title(f"Is conserved transcriptome-orthogonal covariation an ionisation artefact? ({PR.group.iloc[0]}, {len(cc)} samples)", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m13_physchem.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
