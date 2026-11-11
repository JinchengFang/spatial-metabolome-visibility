#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_common as C, m24_external_io as XIO
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200); ap.add_argument("--min-sb", type=float, default=0.7)
ap.add_argument("--max-feat", type=int, default=400, help="number of randomly drawn reliable features for the pruned R2_maldi (one SVD per feature)")
ap.add_argument("--quick", action="store_true", help="registration grid +/-100 µm x 0 degrees only (9 settings)"); ap.add_argument("--step", default="Step23")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); R = a.res


def krr_cv(X, Y, folds, bw_mult=(0.5, 1.0, 2.0), lams=np.logspace(-3, 3, 13)):
    """RBF kernel ridge with spatial-block CV; bandwidth and lambda chosen per output by closed-form leave-one-out error in the training set."""
    X = np.asarray(X, float); Yh = np.empty_like(Y)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xt, Xs = (X[tr] - mu) / sd, (X[te] - mu) / sd
        D2 = ((Xt[:, None, :] - Xt[None, :, :]) ** 2).sum(-1); Ds = ((Xs[:, None, :] - Xt[None, :, :]) ** 2).sum(-1); med = np.median(D2[D2 > 0])
        ym = Y[tr].mean(0); Yt = Y[tr] - ym; best = np.full(Y.shape[1], np.inf); pred = np.zeros((te.sum(), Y.shape[1]))
        for bm in bw_mult:
            g = 1.0 / (bm * med); K = np.exp(-g * D2); Ks = np.exp(-g * Ds); w, Q = np.linalg.eigh(K); QtY = Q.T @ Yt
            for lam in lams * np.mean(w):
                d = w / (w + lam); h = (Q ** 2) @ d
                if h.max() > 0.95:                                                        # near interpolation: LOO formula unreliable (same guard as cv_ridge_gcv)
                    continue
                fit = Q @ (d[:, None] * QtY)
                e = (((Yt - fit) / np.maximum(1 - h, 0.05)[:, None]) ** 2).mean(0); m = e < best
                if m.any():
                    alpha = Q @ ((1.0 / (w + lam))[:, None] * QtY[:, m]); pred[:, m] = Ks @ alpha; best[m] = e[m]
        Yh[te] = pred + ym
    return Yh


def ratio_of(Y, Yh, SB, rel):
    return float(np.nanmedian(C.r2_columns(Y[:, rel], Yh[:, rel]) / SB[rel]))


REG = XIO.load_registration(P); CAP, GRID, RED, V = [], [], [], {}
for name, T in REG.items():
    Sx = XIO.load_section(name, T); Bn = XIO.make_bins(Sx, R, n_pc=100)
    Y, A, B, folds = Bn["Y"], Bn["A"], Bn["B"], Bn["folds"]; _, SB = C.split_half_reliability(A, B); rel = np.where(np.nan_to_num(SB) >= a.min_sb)[0]
    # ---------------------------------------------------------- (1) capacity
    for k in (10, 30, 50, 100):
        if Bn["PC"].shape[1] >= k:
            CAP.append(dict(section=name, model=f"ridge, {k} PCs", n_pred=k, declared=True, ratio_median=ratio_of(Y, C.cv_ridge_gcv(C.zcols(Bn["PC"][:, :k]), Y, folds), SB, rel)))
    for k in (50, 10, 30):                                                                   # 50 = pre-specified; 10 / 30 = post hoc
        CAP.append(dict(section=name, model=f"RBF kernel ridge, {k} PCs", n_pred=k, declared=(k == 50),
                        ratio_median=ratio_of(Y[:, rel], krr_cv(Bn["PC"][:, :k], Y[:, rel], folds), SB[rel], np.arange(len(rel)))))
    cp = pd.DataFrame([c for c in CAP if c["section"] == name]); C.log(f"  {name} capacity: " + "; ".join(f"{r.model} {r.ratio_median:.2f}" for r in cp.itertuples()))
    # ---------------------------------------------------------- (2) registration perturbation
    shifts = (-100, 0, 100) if a.quick else (-200, -100, 0, 100, 200); rots = (0,) if a.quick else (-3, 0, 3)
    for rt in rots:
        for dx in shifts:
            for dy in shifts:
                Bp = XIO.make_bins(Sx, R, shift=(dx, dy), rot_deg=rt, n_pc=30)
                if Bp is None:
                    continue
                _, SBp = C.split_half_reliability(Bp["A"], Bp["B"]); rp = np.where(np.nan_to_num(SBp) >= a.min_sb)[0]
                GRID.append(dict(section=name, dx_um=dx, dy_um=dy, rot_deg=rt, n_bins=Bp["n_bins"], n_reliable=len(rp),
                                 ratio_median=ratio_of(Bp["Y"], C.cv_ridge_gcv(C.zcols(Bp["PC"]), Bp["Y"], Bp["folds"]), SBp, rp)))
    gd = pd.DataFrame([g for g in GRID if g["section"] == name]); b0 = gd[(gd.dx_um == 0) & (gd.dy_um == 0) & (gd.rot_deg == 0)].ratio_median.iloc[0]; bm = gd.loc[gd.ratio_median.idxmax()]
    C.log(f"  {name} registration grid ({len(gd)} settings): original {b0:.2f}; best {bm.ratio_median:.2f} (dx {bm.dx_um}, dy {bm.dy_um}, rotation {bm.rot_deg} deg)")
    # ---------------------------------------------------------- (3) cognate-peak redundancy
    im = Bn["im"]; v = im >= 0; Xp = Sx["X"][v]; ib = im[v]
    Mb, _ = XIO.binmeans(Xp, ib, Bn["n_bins"]); Rp = Xp - Mb[ib]                        # within-bin residual (pixel minus bin mean)
    sub = np.random.default_rng(23).choice(len(Rp), min(20000, len(Rp)), replace=False); Rz = C.zcols(Rp[sub]); Cpix = np.nan_to_num((Rz.T @ Rz) / len(Rz)); np.fill_diagonal(Cpix, 0)
    Cbin = np.nan_to_num((A.T @ A) / len(A)); np.fill_diagonal(Cbin, 0)
    rng = np.random.default_rng(7); feats = rel if len(rel) <= a.max_feat else np.sort(rng.choice(rel, a.max_feat, replace=False))
    r2tx = C.r2_columns(Y, C.cv_ridge_gcv(C.zcols(Bn["PC"][:, :30]), Y, folds))
    for j in feats:
        out = dict(section=name, feature=str(Sx["fid"][j]), SB=SB[j], R2_tx=r2tx[j], max_pixel_resid_r=float(np.abs(Cpix[j]).max()))
        top10 = np.argsort(-np.abs(Cpix[j]))[:10]
        for lab, drop in (("all_others", np.array([j])), ("pruned_pixel", np.union1d(np.union1d(np.where(np.abs(Cpix[j]) > 0.3)[0], top10), [j])),
                          ("pruned_pixel_and_bin0.8", np.union1d(np.union1d(np.union1d(np.where(np.abs(Cpix[j]) > 0.3)[0], top10), np.where(np.abs(Cbin[j]) > 0.8)[0]), [j]))):
            oth = np.setdiff1d(np.arange(Y.shape[1]), drop); Ua, Sa, _ = np.linalg.svd(np.nan_to_num(A[:, oth]), full_matrices=False)
            out[f"R2_maldi_{lab}"] = float(C.r2_columns(B[:, [j]], C.cv_ridge_gcv(Ua[:, :20] * Sa[:20], B[:, [j]], folds))[0]); out[f"n_dropped_{lab}"] = len(drop) - 1
        RED.append(out)
    rd = pd.DataFrame([r for r in RED if r["section"] == name])
    C.log(f"  {name} redundancy pruning ({len(rd)} reliable features): R2_tx {rd.R2_tx.median():.2f}; R2_maldi all {rd.R2_maldi_all_others.median():.2f} -> pixel-pruned {rd.R2_maldi_pruned_pixel.median():.2f} "
          f"({rd.n_dropped_pruned_pixel.mean():.0f} removed on average) -> also bin |r|>0.8 {rd['R2_maldi_pruned_pixel_and_bin0.8'].median():.2f}")
    cd = cp[cp.declared]
    V[name] = dict(capacity_best=float(cd.ratio_median.max()), capacity_ridge30=float(cp[cp.model == "ridge, 30 PCs"].ratio_median.iloc[0]), capacity_best_model=cd.loc[cd.ratio_median.idxmax(), "model"],
                   capacity_best_all=float(cp.ratio_median.max()), capacity_best_all_model=cp.loc[cp.ratio_median.idxmax(), "model"],
                   registration_original=float(b0), registration_best=float(bm.ratio_median), registration_best_at=dict(dx=int(bm.dx_um), dy=int(bm.dy_um), rot=int(bm.rot_deg)),
                   R2_tx=float(rd.R2_tx.median()), R2_maldi_all=float(rd.R2_maldi_all_others.median()), R2_maldi_pruned=float(rd.R2_maldi_pruned_pixel.median()),
                   R2_maldi_pruned_strict=float(rd["R2_maldi_pruned_pixel_and_bin0.8"].median()))
pd.DataFrame(CAP).to_csv(os.path.join(TAB, "m24x_capacity.tsv"), sep="\t", index=False); pd.DataFrame(GRID).to_csv(os.path.join(TAB, "m24x_registration_grid.tsv"), sep="\t", index=False)
pd.DataFrame(RED).to_csv(os.path.join(TAB, "m24x_redundancy.tsv"), sep="\t", index=False)
verdict = dict(per_section=V)
if V:
    verdict["D67a_capacity_does_not_close_gap"] = bool(all(x["capacity_best"] < 0.75 and x["capacity_best"] - x["capacity_ridge30"] < 0.10 for x in V.values()))
    verdict["D67a_posthoc_all_models_below_0.75"] = bool(all(x["capacity_best_all"] < 0.75 for x in V.values()))   # post hoc: includes kernel ridge with 10 / 30 PCs
    verdict["D67b_registration_does_not_close_gap"] = bool(all(x["registration_best"] < 0.75 for x in V.values()))
    verdict["D67c_internal_reference_not_redundancy"] = bool(all(x["R2_maldi_pruned"] > x["R2_tx"] for x in V.values()))
    verdict["D67c_strict_sensitivity"] = bool(all(x["R2_maldi_pruned_strict"] > x["R2_tx"] for x in V.values()))
C.save_json(verdict, os.path.join(TAB, "m24x_verdict.json")); C.log("Criteria: " + str({k: verdict[k] for k in verdict if k.startswith("D67")}))
# ---------------------------------------------------------------- figure
if V:
    secs = list(V); fig, axs = plt.subplots(1, 3, figsize=(11, 3.2)); cap = pd.DataFrame(CAP); grid = pd.DataFrame(GRID); red = pd.DataFrame(RED); cols = ["#3182bd", "#e6550d"]
    models = list(dict.fromkeys(cap.model)); x = np.arange(len(models))
    for i, s_ in enumerate(secs):
        g = cap[cap.section == s_].set_index("model").reindex(models); axs[0].bar(x + (i - 0.5) * 0.38, g.ratio_median, 0.38, color=cols[i % 2], label=s_)
    axs[0].axhline(0.75, c="k", ls="--", lw=.6); axs[0].set_xticks(x); axs[0].set_xticklabels(models, rotation=30, ha="right", fontsize=5.5); axs[0].set_ylabel("median R²_tx / SB"); axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize=6, frameon=False); axs[0].set_title("(1) model capacity", fontsize=7)
    for i, s_ in enumerate(secs):
        g = grid[grid.section == s_]; dist = np.hypot(g.dx_um, g.dy_um); axs[1].scatter(dist + (i - 0.5) * 8, g.ratio_median, s=8 + 6 * np.abs(g.rot_deg), c=cols[i % 2], alpha=.7, label=s_)
    axs[1].axhline(0.75, c="k", ls="--", lw=.6); axs[1].set_xlabel("extra MALDI shift (µm); marker size = |rotation|"); axs[1].set_ylabel("median R²_tx / SB"); axs[1].set_ylim(0, 1)
    axs[1].set_title("(2) registration perturbation (75 per section)", fontsize=7); axs[1].legend(fontsize=6, frameon=False)
    labs = [("R2_tx", "transcriptome"), ("R2_maldi_all_others", "MALDI: all other features"), ("R2_maldi_pruned_pixel", "MALDI: homologous peaks removed"), ("R2_maldi_pruned_pixel_and_bin0.8", "+ bin-level |r| > 0.8 removed")]
    x = np.arange(len(labs))
    for i, s_ in enumerate(secs):
        g = red[red.section == s_]; axs[2].bar(x + (i - 0.5) * 0.38, [g[c].median() for c, _ in labs], 0.38, color=cols[i % 2], label=s_)
    axs[2].set_xticks(x); axs[2].set_xticklabels([l for _, l in labs], rotation=30, ha="right", fontsize=5.5); axs[2].set_ylabel("median cross-validated R²"); axs[2].legend(fontsize=6, frameon=False)
    axs[2].set_title("(3) internal reference without homologous peaks", fontsize=7)
    fig.suptitle("External dataset: robustness of the partial-explanation result (200 µm, reliable features)", fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "m24x_robustness.png"), dpi=180); plt.close(fig)
C.log(f"Done ({time.time() - t0:.0f} s)")
