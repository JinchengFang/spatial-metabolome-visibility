#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="SMA,ccRCC,GBM,IDHmut,EXT")
ap.add_argument("--res", type=int, default=200); ap.add_argument("--res-ext", default="200,400"); ap.add_argument("--n-sim", type=int, default=40)
ap.add_argument("--min-sb", type=float, default=0.7); ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--min-rel", type=int, default=10)
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--min-px", type=int, default=2); ap.add_argument("--ccrcc-dir", default=None)
ap.add_argument("--ion-set", default="strict"); ap.add_argument("--step", default="Step23")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
DS = [d.strip() for d in a.datasets.split(",") if d.strip()]


def colcorr(E1, E2):
    return (C.zcols(E1) * C.zcols(E2)).mean(0)


def insample_ridge(X, Y, alphas=np.logspace(-2, 6, 17)):
    """In-sample ridge fit (standardised X; GCV alpha per column; effective df <= 0.9 n)."""
    X = np.asarray(X, float); X = X[:, X.std(0) > 1e-9]; Xs = C.zcols(X); Yc = Y - Y.mean(0); n = len(Y)
    U, s, _ = np.linalg.svd(Xs, full_matrices=False); UtY = U.T @ Yc; best = np.full(Y.shape[1], np.inf); fit = np.zeros_like(Yc)
    for al in alphas:
        d = s ** 2 / (s ** 2 + al); df = d.sum()
        if df > 0.9 * n:
            continue
        f_ = U @ (d[:, None] * UtY); g = ((Yc - f_) ** 2).mean(0) / (1 - df / n) ** 2; m = g < best; best[m] = g[m]; fit[:, m] = f_[:, m]
    return fit


def est_internal(X, Y, A, B, folds):
    """M12-01 estimator: full-data R2; split-half correlation of residuals after fitting A and B separately."""
    r2 = C.r2_columns(Y, C.cv_ridge_gcv(X, Y, folds)); _, SB = C.split_half_reliability(A, B)
    return r2, SB, colcorr(A - C.cv_ridge_gcv(X, A, folds), B - C.cv_ridge_gcv(X, B, folds))


def est_external(X, Y, A, B, folds):
    """M23-03 estimator: 30 gene PCs; A and B minus the same full-data prediction."""
    Yh = C.cv_ridge_gcv(C.zcols(X), Y, folds); _, SB = C.split_half_reliability(A, B)
    return C.r2_columns(Y, Yh), SB, colcorr(A - Yh, B - Yh)


def projection(X, Y):
    X = np.asarray(X, float); X = C.zcols(X[:, X.std(0) > 1e-9]); Yc = Y - Y.mean(0)
    return X @ np.linalg.lstsq(X, Yc, rcond=None)[0]


def oracle(X, Y, A, B, folds, est, n_sim, rng, kind="ridge"):
    rho = np.clip((A * B).mean(0), 0.05, 0.98); Sg = C.zcols(insample_ridge(X, Y) if kind == "ridge" else projection(X, Y)); out = []
    for _ in range(n_sim):
        As = np.sqrt(rho) * Sg + np.sqrt(1 - rho) * rng.standard_normal(Sg.shape); Bs = np.sqrt(rho) * Sg + np.sqrt(1 - rho) * rng.standard_normal(Sg.shape)
        out.append(est(X, C.zcols(As + Bs), C.zcols(As), C.zcols(Bs), folds))
    r2s, sbs, res = (np.array([o[i] for o in out]) for i in range(3))
    return r2s, sbs, res


FR, SR = [], []


def run_sample(dataset, sample, design, R, X, Y, A, B, folds, est, rng):
    r2, SB, rE = est(X, Y, A, B, folds); rel = np.where(np.nan_to_num(SB) >= a.min_sb)[0]
    if len(rel) < a.min_rel:
        C.log(f"  {dataset} {sample} @ {R} µm: {len(rel)} reliable features < {a.min_rel}, skipped"); return
    ratio = r2[rel] / SB[rel]; obs = float(np.nanmedian(rE[rel])); res_k = {}
    for kind in ("ridge", "projection"):
        r2s, sbs, res = oracle(X, Y[:, rel], A[:, rel], B[:, rel], folds, est, a.n_sim, rng, kind)
        att = np.nanmedian(r2s / np.where(sbs > 0, sbs, np.nan), 0); null_med = np.nanmedian(res, 1)
        res_k[kind] = dict(att=att, null_med=null_med, null_feat=np.nanmedian(res, 0))
    att = np.minimum(res_k["ridge"]["att"], res_k["projection"]["att"])                  # conservative: smaller ceiling
    cal = ratio / np.maximum(att, 0.05)
    q95 = {k: float(np.nanpercentile(res_k[k]["null_med"], 95)) for k in res_k}; kq = max(q95, key=q95.get)  # conservative: higher null
    for k, j in enumerate(rel):
        FR.append(dict(dataset=dataset, sample=sample, design=design, res_um=R, feature=j, SB=SB[j], R2_tx=r2[j], ratio=ratio[k], attainable_ridge=res_k["ridge"]["att"][k],
                       attainable_projection=res_k["projection"]["att"][k], attainable=att[k], calibrated=cal[k], r_resid=rE[j],
                       r_resid_null_ridge=res_k["ridge"]["null_feat"][k], r_resid_null_projection=res_k["projection"]["null_feat"][k]))
    SR.append(dict(dataset=dataset, sample=sample, design=design, res_um=R, n_bins=len(Y), n_pred=X.shape[1], n_reliable=len(rel), ratio_median=float(np.nanmedian(ratio)),
                   attainable_ridge_median=float(np.nanmedian(res_k["ridge"]["att"])), attainable_projection_median=float(np.nanmedian(res_k["projection"]["att"])),
                   attainable_median=float(np.nanmedian(att)), calibrated_median=float(np.nanmedian(cal)), r_resid_median=obs,
                   r_resid_null_median=float(np.nanmedian(res_k[kq]["null_med"])), r_resid_null_q95=q95[kq], null_q95_ridge=q95["ridge"], null_q95_projection=q95["projection"],
                   p_r_resid=float((1 + (res_k[kq]["null_med"] >= obs).sum()) / (1 + a.n_sim))))
    s = SR[-1]
    C.log(f"  {dataset} {sample} @ {R} µm ({design}): {len(Y)} bins, {X.shape[1]} predictors, {len(rel)} reliable features | R2_tx/SB {s['ratio_median']:.2f}, "
          f"attainable ridge {s['attainable_ridge_median']:.2f} / projection {s['attainable_projection_median']:.2f} -> calibrated {s['calibrated_median']:.2f} | "
          f"r_E {obs:.2f} vs null 95% ridge {q95['ridge']:.2f} / projection {q95['projection']:.2f}")


# ------------------------------------------------------------------ internal data (as in M12-01)
INT = [d for d in DS if d != "EXT"]
if INT:
    import p2_loaders as L
    cfg = L.Config(P, ion_set=a.ion_set, ccrcc_dir=a.ccrcc_dir); SHIFT = {}
    fs = os.path.join(P["P2"], "05_tables", "Step8", "m11_registration_summary.tsv")
    if os.path.exists(fs):
        rs = pd.read_csv(fs, sep="\t"); rs = rs[(rs.res_um == rs.res_um.min()) & (rs.cv_gain > 0.02)]
        SHIFT = {r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples()}
    for S in L.load_all(cfg, INT):
        R = a.res; G = L.make_grid(S, R, cfg)
        if G is None:
            continue
        sh = SHIFT.get(S["sample"], (0.0, 0.0))
        Y, c0 = L.pixel_means(S, G, None, sh); YA, cA = L.pixel_means(S, G, S["px_half"] == 0, sh); YB, cB = L.pixel_means(S, G, S["px_half"] == 1, sh)
        ok = (c0 >= a.min_px) & (cA >= a.min_px) & (cB >= a.min_px)
        Bk, _, _ = L.tx_blocks(S, G["CB"], G["ncell"], G["comp_fixed"])
        X = np.column_stack([np.asarray(v, float) for v in Bk.values()])[ok]; X = X[:, X.std(0) > 1e-9]
        Y, YA, YB, xy = Y[ok], YA[ok], YB[ok], G["xy"][ok]
        det = (np.expm1(Y) > 0).mean(0) if S["feat_group"] is None else np.ones(Y.shape[1]); okf = np.where(det >= a.min_det)[0]
        design = ("same-section" if S.get("same_section") else "adjacent-section (same mouse, artificial)" if S["dataset"] == "SMA-cross"
                  else "adjacent-section" if S["dataset"] in ("ccRCC", "GBM") else "registered-cells")          # SMA-cross: artificial adjacent pairs, same mouse
        run_sample(S["dataset"], S["sample"], design, R, X, C.zcols(Y[:, okf]), C.zcols(YA[:, okf]), C.zcols(YB[:, okf]),
                   C.spatial_folds(xy, a.block_um, seed=R), est_internal, np.random.default_rng(23))

# ------------------------------------------------------------------ external data (as in M23-03)
if "EXT" in DS:
    import m24_external_io as XIO
    REG = XIO.load_registration(P)
    for name, T in REG.items():
        Sx = XIO.load_section(name, T, a.min_det)
        for R in [int(r) for r in a.res_ext.split(",")]:
            Bn = XIO.make_bins(Sx, R, n_pc=30)
            if Bn is None:
                continue
            run_sample("External-COSIE", name, "external (another lab)", R, Bn["PC"], Bn["Y"], Bn["A"], Bn["B"], Bn["folds"], est_external, np.random.default_rng(23))

F, S = pd.DataFrame(FR), pd.DataFrame(SR)
if not len(S):
    raise SystemExit("No samples to analyse")
F.to_csv(os.path.join(TAB, "m24_oracle_features.tsv"), sep="\t", index=False); S.to_csv(os.path.join(TAB, "m24_oracle_summary.tsv"), sep="\t", index=False)
ext2, ext4, inn = S[(S.dataset == "External-COSIE") & (S.res_um == 200)], S[(S.dataset == "External-COSIE") & (S.res_um == 400)], S[S.dataset != "External-COSIE"]
v = dict(n_internal=len(inn), n_external_200=len(ext2))
v["internal_frac_calibrated_lt_0.75"] = float((inn.calibrated_median < 0.75).mean()) if len(inn) else None
v["internal_frac_rE_gt_null95"] = float((inn.r_resid_median > inn.r_resid_null_q95).mean()) if len(inn) else None
if len(ext2):
    v["D66a_gap_not_attenuation"] = bool((ext2.calibrated_median < 0.75).all() and (not len(inn) or v["internal_frac_calibrated_lt_0.75"] >= 0.75))
    v["D66b_residual_not_shared_prediction"] = bool((ext2.r_resid_median > ext2.r_resid_null_q95).all() and (not len(inn) or v["internal_frac_rE_gt_null95"] >= 0.75))
else:
    v["D66a_gap_not_attenuation"] = v["D66b_residual_not_shared_prediction"] = "not testable (no external data)"
if len(ext2) and len(ext4):
    mg = ext2.merge(ext4, on="sample", suffixes=("_200", "_400"))
    v["D66c_400um_is_attenuation"] = bool((mg.attainable_median_400 < 0.6).all() and ((mg.calibrated_median_400 - mg.calibrated_median_200).abs() < 0.25).all())
else:
    v["D66c_400um_is_attenuation"] = "not testable"
for d_ in ("same-section", "adjacent-section (same mouse, artificial)", "adjacent-section", "registered-cells", "external (another lab)"):
    g = S[(S.design == d_) & (S.res_um == 200)]
    if len(g):
        v[f"median_{d_}"] = dict(ratio=float(g.ratio_median.median()), attainable=float(g.attainable_median.median()), calibrated=float(g.calibrated_median.median()), n=len(g))
v["summary"] = S.round(3).to_dict("records")
C.save_json(v, os.path.join(TAB, "m24_oracle_verdict.json"))
C.log("Summary:\n" + S[["dataset", "sample", "res_um", "n_bins", "n_reliable", "ratio_median", "attainable_median", "calibrated_median", "r_resid_median", "r_resid_null_q95"]].round(2).to_string(index=False) + "\nCriteria: " + str({k: v[k] for k in v if k.startswith("D66") or k.startswith("internal_") or k.startswith("median_")}))
# ------------------------------------------------------------------ figure
S2 = S.copy(); S2["label"] = S2["sample"].astype(str).str.replace("v11l12-", "", regex=False).str.replace("v11t17-", "", regex=False) + np.where(S2.res_um != 200, " @" + S2.res_um.astype(str), "")
order = ["same-section", "adjacent-section (same mouse, artificial)", "registered-cells", "adjacent-section", "external (another lab)"]; S2["o"] = S2.design.map({d: i for i, d in enumerate(order)}); S2 = S2.sort_values(["o", "dataset", "label"])
x = np.arange(len(S2)); fig, axs = plt.subplots(2, 1, figsize=(max(7, 0.32 * len(S2) + 2), 6), sharex=True)
axs[0].bar(x - 0.27, S2.ratio_median, 0.27, color="#9ecae1", label="observed R²_tx / SB"); axs[0].bar(x, S2.attainable_median, 0.27, color="#bdbdbd", label="attainable under perfect transcriptomic determination")
axs[0].bar(x + 0.27, S2.calibrated_median, 0.27, color="#08519c", label="calibrated (observed ÷ attainable)"); axs[0].axhline(0.75, c="k", ls="--", lw=.6); axs[0].axhline(1, c="#999", lw=.4)
axs[0].set_ylabel("median over reliable features"); axs[0].legend(fontsize=5.5, frameon=False, ncol=3, loc="upper left"); axs[0].set_ylim(0, 1.3)
axs[1].errorbar(x, S2.r_resid_null_median, yerr=[S2.r_resid_null_median - S2.r_resid_null_median, S2.r_resid_null_q95 - S2.r_resid_null_median], fmt="o", c="#999", ms=3, label="null (perfect determination): median, 95th pct")
axs[1].scatter(x, S2.r_resid_median, c="#cb181d", s=14, zorder=3, label="observed"); axs[1].set_ylabel("split-half r of transcriptome-orthogonal residual"); axs[1].legend(fontsize=5.5, frameon=False)
axs[1].set_xticks(x); axs[1].set_xticklabels(S2.label, rotation=70, ha="right", fontsize=5.5)
for ax_ in axs:
    for b in np.where(np.diff(S2.o.values) != 0)[0]:
        ax_.axvline(b + 0.5, c="#ccc", lw=.5)
fig.suptitle("Oracle calibration: how much of the reliable metabolite variance could the same estimator recover if the transcriptome determined it completely?", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m24_oracle.png"), dpi=180); plt.close(fig)
C.log(f"Done ({time.time() - t0:.0f} s)")
