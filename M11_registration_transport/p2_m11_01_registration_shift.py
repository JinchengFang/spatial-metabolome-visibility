#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="IDHmut,ccRCC,GBM"); ap.add_argument("--res", default="200,400")
ap.add_argument("--ccrcc-dir", default=None); ap.add_argument("--ion-set", default="strict"); ap.add_argument("--include-uncertain", action="store_true")
ap.add_argument("--max-feat", type=int, default=60); ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--cluster-r", type=float, default=0.9)
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--deltas", default="0,50,100,200,400")
ap.add_argument("--grid-span", type=float, default=300.0); ap.add_argument("--grid-step", type=float, default=50.0); ap.add_argument("--min-px", type=int, default=2)
ap.add_argument("--step", default="Step8", help="output folder (same-section data are run with --step Step10 --datasets SMA)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
cfg = L.Config(P, ion_set=a.ion_set, ccrcc_dir=a.ccrcc_dir, include_uncertain=a.include_uncertain)
RES = [int(r) for r in a.res.split(",")]; DS = [d.strip() for d in a.datasets.split(",") if d.strip()]
DEL = [float(x) for x in a.deltas.split(",")]; t0 = time.time(); plt.rcParams.update({"font.size": 7})


def r2(Y, X, folds):
    return C.r2_columns(Y, C.cv_ridge_gcv(X, Y, folds))


curves, grid_rows, summ = [], [], []
for S in L.load_all(cfg, DS):
    for R in RES:
        G = L.make_grid(S, R, cfg)
        if G is None:
            C.log(f"  {S['dataset']} {S['sample']} @ {R} µm: too few bins; skipped"); continue
        Y0, cnt0 = L.pixel_means(S, G)
        det = (np.expm1(Y0) > 0).mean(0) if S["feat_group"] is None else np.ones(Y0.shape[1])
        okf = np.where(det >= a.min_det)[0]
        reps, I, q, _ = C.select_structured_features(C.zcols(Y0[:, okf]), G["xy"], a.max_feat, a.cluster_r, seed=R)
        if len(reps) < 3:
            continue
        cols = okf[reps]
        Bk, LG, src = L.tx_blocks(S, G["CB"], G["ncell"], G["comp_fixed"])
        X0 = np.column_stack([np.asarray(v, float) for v in Bk.values()]); X0 = X0[:, X0.std(0) > 1e-9]
        A0, cA = L.pixel_means(S, G, S["px_half"] == 0)
        folds_all = C.spatial_folds(G["xy"], a.block_um, seed=R); T_all = C.poly2(G["xy"])
        C.log(f"  {S['dataset']} {S['sample']} @ {R} µm: {len(G['xy'])} bins, {len(cols)} metabolite features; shifts {DEL} µm x 8 directions + 2-D grid")

        B00, cB0 = L.pixel_means(S, G, S["px_half"] == 1)

        def metrics(Ys, Bs, ok, lv, do_maldi):
            T = T_all[ok]
            prep = (lambda M_: C.zcols(M_)) if lv == "full" else (lambda M_: C.zcols(C.detrend(M_, T)))
            Yt, Bt_, At = prep(Ys[ok][:, cols]), prep(Bs[ok][:, cols]), prep(A0[ok][:, cols])
            Xl = X0[ok] if lv == "full" else C.detrend(X0[ok], T); Xl = Xl[:, Xl.std(0) > 1e-9]
            f = folds_all[ok]; res = dict(tx=np.median(r2(Yt, Xl, f)))
            if do_maldi:
                res["self"] = np.median([r2(Bt_[:, [j]], At[:, [j]], f)[0] for j in range(len(cols))])
                mal = []
                for j in range(len(cols)):
                    oth = np.delete(At, j, axis=1); Op = PCA(min(15, oth.shape[1]), random_state=0).fit_transform(oth)
                    mal.append(r2(Bt_[:, [j]], Op, f)[0])
                res["maldi"] = np.median(mal)
            return res

        def evaluate(shift, levels=("full", "detrended"), do_maldi=True, sub=None):
            """Shifted vs delta = 0 on the same bins (avoids bias from edge bins). Returns {level: {metric: (R² shifted, R² delta 0)}}."""
            Ys, cs = L.pixel_means(S, G, None, shift); Bs, cB = L.pixel_means(S, G, S["px_half"] == 1, shift)
            ok = (cs >= a.min_px) & (cB >= a.min_px) & (cA >= a.min_px) & (cnt0 >= a.min_px) & (cB0 >= a.min_px)
            if sub is not None:
                ok &= sub
            if ok.sum() < cfg.min_bins // 2:
                return None
            out = {}
            for lv in levels:
                m1 = metrics(Ys, Bs, ok, lv, do_maldi)
                m0 = metrics(Y0, B00, ok, lv, do_maldi) if any(shift) else m1
                out[lv] = {k: (m1[k], m0[k]) for k in m1}
            return out

        b0 = evaluate((0.0, 0.0)); base = {lv: {k: v[0] for k, v in r_.items()} for lv, r_ in b0.items()}
        for d in DEL:
            dirs = [(0.0, 0.0)] if d == 0 else [(d * np.cos(t), d * np.sin(t)) for t in np.arange(8) * np.pi / 4]
            acc = {}
            for sh in dirs:
                e = evaluate(sh)
                if e is None:
                    continue
                for lv, r_ in e.items():
                    for k, (v1, v0) in r_.items():
                        acc.setdefault((lv, k), []).append((v1, v0))
            for (lv, k), v in acc.items():
                v = np.array(v); m1, m0 = v[:, 0].mean(), v[:, 1].mean()
                curves.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=R, level=lv, metric=k, delta_um=d, R2=float(m1), R2_paired0=float(m0),
                                   retention=float(m1 / m0) if m0 > 0.01 else np.nan))
        # 2-D grid: R²_tx at the full level only; gain = shifted - delta 0 on the same bins.
        # the maximum over 169 shifts is biased -> cross-fitting: 1-mm blocks in two checkerboard groups; shift chosen on one, gain evaluated on the other (and vice versa)
        best = (0.0, 0.0, 0.0, base["full"]["tx"]); cv_gain = np.nan; cv_shift = (np.nan, np.nan)
        if R == RES[0]:
            bx = np.floor(G["xy"] / a.block_um).astype(int); grp = (bx[:, 0] + bx[:, 1]) % 2
            span = np.arange(-a.grid_span, a.grid_span + 1, a.grid_step); gains = {0: {}, 1: {}}
            for dx in span:
                for dy in span:
                    e = evaluate((dx, dy), levels=("full",), do_maldi=False)
                    if e is None:
                        continue
                    v1, v0 = e["full"]["tx"]
                    row_ = dict(dataset=S["dataset"], sample=S["sample"], res_um=R, dx=dx, dy=dy, R2_tx=v1, R2_tx_paired0=v0, gain=v1 - v0)
                    for g_ in (0, 1):
                        eg = evaluate((dx, dy), levels=("full",), do_maldi=False, sub=grp == g_)
                        if eg is not None:
                            gains[g_][(dx, dy)] = eg["full"]["tx"][0] - eg["full"]["tx"][1]; row_[f"gain_half{g_}"] = gains[g_][(dx, dy)]
                    grid_rows.append(row_)
                    if v1 - v0 > best[0]:
                        best = (v1 - v0, dx, dy, v1)
            if gains[0] and gains[1]:
                k0 = max(gains[0], key=gains[0].get); k1 = max(gains[1], key=gains[1].get)
                cv_gain = 0.5 * (gains[1].get(k0, 0.0) + gains[0].get(k1, 0.0)); cv_shift = k0
        rA, SBv = C.split_half_reliability(A0[:, cols], B00[:, cols]); SB_med = float(np.nanmedian(SBv))
        cv = pd.DataFrame([c for c in curves if c["sample"] == S["sample"] and c["res_um"] == R])
        def ret(lv, k, d):
            v = cv[(cv.level == lv) & (cv.metric == k) & (cv.delta_um == d)].retention
            return float(v.iloc[0]) if len(v) else np.nan
        row = dict(dataset=S["dataset"], sample=S["sample"], res_um=R, n_bins=len(G["xy"]), n_feat=len(cols),
                   R2_tx_0=base["full"]["tx"], R2_self_0=base["full"]["self"], R2_maldi_0=base["full"]["maldi"],
                   best_dx=best[1], best_dy=best[2], best_shift_um=float(np.hypot(best[1], best[2])), best_gain=best[0], R2_tx_best=best[3],
                   **{f"ret_{k}_{int(d)}": ret("full", k, d) for k in ("tx", "self", "maldi") for d in (50, 100, 200)},
                   **{f"ret_det_{k}_100": ret("detrended", k, 100) for k in ("tx", "self", "maldi")})
        row.update(cv_gain=cv_gain, cv_shift_dx=cv_shift[0], cv_shift_dy=cv_shift[1], ceiling_SB=SB_med,
                   R2_tx_shift_adjusted=base["full"]["tx"] + max(np.nan_to_num(cv_gain), 0.0))
        row["tx_over_ceiling_0"] = base["full"]["tx"] / SB_med; row["tx_over_ceiling_adjusted"] = row["R2_tx_shift_adjusted"] / SB_med
        # D34: with the best rigid shift (cross-fitted gain = upper bound of residual misalignment) and <= 40% loss of perfect coupling at 100 µm, R²_tx stays < half of the ceiling
        row["D34_misregistration_not_limiting"] = bool((row["tx_over_ceiling_adjusted"] < 0.5) and (row["ret_self_100"] >= 0.6))
        summ.append(row)
        C.log(f"     delta=0: R²_tx {base['full']['tx']:+.3f}, self {base['full']['self']:+.3f}, MALDI {base['full']['maldi']:+.3f}; "
              f"retention at 100 µm tx {row['ret_tx_100']:.2f} / self {row['ret_self_100']:.2f} / MALDI {row['ret_maldi_100']:.2f}; "
              f"best shift ({best[1]:+.0f}, {best[2]:+.0f}) µm, gain {row['best_gain']:+.3f} (cross-fitted {cv_gain:+.3f}); "
              f"R²_tx / ceiling {row['tx_over_ceiling_0']:.2f} -> after shift correction {row['tx_over_ceiling_adjusted']:.2f}")
if not summ:
    raise SystemExit("no results")
CV, GR, SM = pd.DataFrame(curves), pd.DataFrame(grid_rows), pd.DataFrame(summ)
CV.to_csv(os.path.join(TAB, "m11_shift_curves.tsv"), sep="\t", index=False)
GR.to_csv(os.path.join(TAB, "m11_shift_grid.tsv"), sep="\t", index=False)
SM.to_csv(os.path.join(TAB, "m11_registration_summary.tsv"), sep="\t", index=False)
C.log("summary:\n" + SM.round(3).to_string(index=False))

R0 = RES[0]; d = CV[(CV.res_um == R0) & (CV.level == "full")]
keys = d[["dataset", "sample"]].drop_duplicates().values.tolist()
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for k_, (metric, title) in enumerate((("self", "perfect coupling (same metabolite, other half)"), ("maldi", "other metabolites → target"),
                                      ("tx", "transcriptome → metabolite"))):
    for ds, smp in keys:
        e = d[(d.dataset == ds) & (d["sample"] == smp) & (d.metric == metric)].sort_values("delta_um")
        axs[k_].plot(e.delta_um, e.R2, "o-", ms=2.5, lw=.9, label=f"{ds} {smp}")
    axs[k_].set_xlabel("MALDI shift Δ (µm)"); axs[k_].set_ylabel("median cross-validated R²"); axs[k_].set_title(title, fontsize=7)
    axs[k_].axvspan(0, 100, color="#fee08b", alpha=.3, lw=0)
axs[0].legend(fontsize=5, frameon=False)
fig.suptitle(f"Registration sensitivity ({R0} µm bins): shaded = plausible adjacent-section / registration error", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FIG, "m11_registration.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
