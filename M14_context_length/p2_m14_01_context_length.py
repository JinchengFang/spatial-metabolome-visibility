#!/usr/bin/env python3

import argparse, os, re, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="IDHmut,ccRCC,GBM,SMA")
ap.add_argument("--res", type=int, default=200); ap.add_argument("--res-sma", type=int, default=200)
ap.add_argument("--sigmas", default="0,100,200,400,800,1600"); ap.add_argument("--max-feat", type=int, default=100); ap.add_argument("--n-gene", type=int, default=200)
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--cluster-r", type=float, default=0.9)
ap.add_argument("--ccrcc-dir", default=None); ap.add_argument("--ion-set", default="strict"); ap.add_argument("--include-uncertain", action="store_true")
ap.add_argument("--no-shift", action="store_true"); ap.add_argument("--keep-inorganic", action="store_true"); ap.add_argument("--jitter-um", type=float, default=200.0); ap.add_argument("--step", default="Step11"); ap.add_argument("--min-px", type=int, default=1)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
cfg = L.Config(P, ion_set=a.ion_set, ccrcc_dir=a.ccrcc_dir, include_uncertain=a.include_uncertain, min_px_half=1, min_bins=80)
SIG = [float(x) for x in a.sigmas.split(",")]; DS = [d.strip() for d in a.datasets.split(",") if d.strip()]; rng = np.random.default_rng(0)
SHIFT = {}
for st_ in ("Step8", "Step10", "Step11"):      # cross-fitted shifts of adjacent sections (Step 8), same-section SMA (Step 10) and SMA cross pairs (Step 11)
    fs = os.path.join(P["P2"], "05_tables", st_, "m11_registration_summary.tsv")
    if os.path.exists(fs) and not a.no_shift:
        rs = pd.read_csv(fs, sep="\t"); rs = rs[(rs.res_um == rs.res_um.min()) & (rs.cv_gain > 0.02)]
        SHIFT.update({r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples()})
MODS = None
fm = os.path.join(P["P2"], "05_tables", "Step9", "m12_module_members.tsv")
if os.path.exists(fm):
    MODS = pd.read_csv(fm, sep="\t"); MODS["feature"] = MODS.feature.astype(str)


def chem_class(mz, ann):
    first = str(ann).split(";")[0].strip()
    if re.match(r"^(PC|PE|PS|PI|PA|PG|SM|Cer|HexCer|ST|CE|TG|DG|SHexCer|LPC|LPE|LPI|LPG|LPS)\s", first):
        return "lipid"
    if not np.isfinite(mz):
        return "small" if first else "unknown"
    return "small" if mz < 400 else ("lipid" if mz > 600 else "mid")


def kernel(xy, s):
    if s <= 0:
        return None
    tree = cKDTree(xy); pairs = tree.sparse_distance_matrix(tree, 3 * s, output_type="coo_matrix")
    w = np.exp(-0.5 * (pairs.data / s) ** 2); K = sp.csr_matrix((w, (pairs.row, pairs.col)), shape=(len(xy), len(xy))) + sp.eye(len(xy))
    return sp.diags(1 / np.asarray(K.sum(1)).ravel()) @ K


def smooth_of(Y, nbr):
    """Spatial smoothness of the target = correlation with the mean of its 6 nearest neighbours (sigma* may only reflect that smooth targets favour smooth predictors)."""
    Ym = Y[nbr].mean(1); a_, b_ = Y - Y.mean(0), Ym - Ym.mean(0)
    return (a_ * b_).sum(0) / np.sqrt((a_ ** 2).sum(0) * (b_ ** 2).sum(0) + 1e-12)


def adj_test(A, B):
    """OLS log2(sigma*+50) ~ 1 + [in A] + target smoothness. Returns (coefficient of A, one-sided p). A/B: DataFrames with sigma_star and smooth."""
    from scipy.stats import t as tdist
    if len(A) < 3 or len(B) < 3:
        return np.nan, np.nan
    y = np.log2(np.r_[A.sigma_star, B.sigma_star] + 50); X = np.column_stack([np.ones(len(y)), np.r_[np.ones(len(A)), np.zeros(len(B))], np.r_[A.smooth, B.smooth]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta; dof = len(y) - 3
    if dof < 1:
        return np.nan, np.nan
    se = np.sqrt((res @ res) / dof * np.linalg.pinv(X.T @ X)[1, 1])
    return float(beta[1]), float(tdist.sf(beta[1] / max(se, 1e-12), dof))


def r2cv(X, Y, folds):
    return C.r2_columns(Y, C.cv_ridge_gcv(X, Y, folds))


frows, crows, srows = [], [], []
for S in L.load_all(cfg, DS):
    R = a.res_sma if S["dataset"].startswith("SMA") else a.res
    G = L.make_grid(S, R, cfg)
    if G is None:
        C.log(f"  {S['dataset']} {S['sample']} @ {R} µm: too few bins; skipped"); continue
    sh = SHIFT.get(S["sample"], (0.0, 0.0))
    Y0, cnt = L.pixel_means(S, G, None, sh); ok = cnt >= a.min_px
    det = (np.expm1(Y0[ok]) > 0).mean(0) if S["feat_group"] is None else np.ones(Y0.shape[1])
    okf = np.where(det >= a.min_det)[0]; xy = G["xy"][ok]; n_inorg = 0
    if S["mz"] is not None and not a.keep_inorganic:      # remove peaks with non-organic mass defect (salts / inorganic clusters / matrix-related)
        good = C.organic_mass_defect_ok(np.asarray(S["mz"], float)[okf]); n_inorg = int((~good).sum()); okf = okf[good]
    reps, I, q, _ = C.select_structured_features(C.zcols(Y0[ok][:, okf]), xy, a.max_feat, a.cluster_r, seed=R)
    if len(reps) < 5:
        C.log(f"  {S['dataset']} {S['sample']}: fewer than 5 structured features; skipped"); continue
    cols = okf[reps]; names = np.asarray(S["names"]).astype(str)[cols]
    mz = np.asarray(S["mz"], float)[cols] if S["mz"] is not None else np.full(len(cols), np.nan)
    pol = (np.asarray(S["pol"], dtype=object)[cols] if not isinstance(S["pol"], str) else np.array([S["pol"]] * len(cols))).astype(str)
    ann = names if S["annotated"] else np.array([C.annotate_mz(m_, p_, cfg.LM) for m_, p_ in zip(mz, pol)], dtype=object).astype(str)
    cls = np.array([chem_class(m_, an) for m_, an in zip(mz, ann)])
    Ym = Y0[ok][:, cols]
    # module eigengenes (ccRCC, conserved orthogonal modules of Step 9)
    mod_targets = {}
    if S["dataset"] == "ccRCC" and MODS is not None:
        nm_all = np.asarray(S["names"]).astype(str)
        for m_, g in MODS.groupby("module"):
            j = [np.where(nm_all == f)[0][0] for f in g.feature if f in set(nm_all)]
            if len(j) >= 3:
                Zm = C.zcols(Y0[ok][:, j]); U, S_, Vt = np.linalg.svd(Zm, full_matrices=False); mod_targets[m_] = U[:, 0] * np.sign(Vt[0].sum() or 1)
    # transcriptomic predictors (all genes) and gene baseline (n_gene held-out HVGs predicted from the remaining genes)
    CB = G["CB"][np.where(ok)[0]]; Bk, LG, src = L.tx_blocks(S, CB, G["ncell"][ok], None if G["comp_fixed"] is None else G["comp_fixed"][ok])
    X0 = np.column_stack([np.asarray(v, float) for v in Bk.values()]); X0 = X0[:, X0.std(0) > 1e-9]
    gv = LG.var(0); cand = np.argsort(-gv)[: max(2000, a.n_gene * 5)]; cand = cand[gv[cand] > 0]
    ho = np.sort(rng.choice(cand, min(a.n_gene, len(cand) // 3), replace=False)); keep = np.setdiff1d(np.arange(CB.shape[1]), ho)
    CA, CH = C.thin_counts(CB, rng)
    Bg, _, _ = C.tx_blocks_from_bins(CA[:, keep], np.asarray(S["genes"])[keep], sig=S["sig"], programs_fallback=S["programs"], state_sets=S["state_sets"],
                                     state_min_genes=S["state_min"], comp_fixed=None if G["comp_fixed"] is None else G["comp_fixed"][ok],
                                     tech_extra=np.log1p(G["ncell"][ok]) if S["kind"] == "xenium" else None)
    Xg = np.column_stack([np.asarray(v, float) for v in Bg.values()]); Xg = Xg[:, Xg.std(0) > 1e-9]
    totH = np.asarray(CH.sum(1)).ravel(); Tg = np.log1p((sp.diags(1e4 / np.maximum(totH, 1)) @ CH[:, ho]).toarray()); gnames = np.asarray(S["genes"])[ho]
    # the first 10 PCs of the held-out genes are also targets: single Visium genes are noisy and much less smooth than metabolite maps; PCs make the smoothness of the baseline overlap with metabolites
    Uh, Sh, _ = np.linalg.svd(C.zcols(Tg), full_matrices=False); npc_h = min(10, Tg.shape[1] - 1)
    Tg = np.column_stack([Tg, Uh[:, :npc_h] * Sh[:npc_h]]); gnames = np.r_[gnames, [f"heldout_PC{i + 1}" for i in range(npc_h)]]
    gcls = np.r_[["gene"] * len(ho), ["gene_pc"] * npc_h]
    # registration matching: target from the nearest bin to x + d (|d| = jitter, random direction; nearest bin within 0.75 bin spacings)
    ang = rng.uniform(0, 2 * np.pi); dvec = a.jitter_um * np.array([np.cos(ang), np.sin(ang)])
    dd, jj = cKDTree(xy).query(xy + dvec); jok = np.where(dd < 0.75 * R)[0]; jidx = jj[jok]
    folds_j = C.spatial_folds(xy[jok], a.block_um, seed=R)
    folds = C.spatial_folds(xy, a.block_um, seed=R); T = C.poly2(xy)
    nbr = cKDTree(xy).query(xy, 7)[1][:, 1:]; nbr_j = cKDTree(xy[jok]).query(xy[jok], 7)[1][:, 1:]
    Ks = {s_: kernel(xy, s_) for s_ in SIG}
    C.log(f"  {S['dataset']} {S['sample']} @ {R} µm{' (shift ' + str(sh) + ')' if any(sh) else ''}: {len(xy)} bins, {len(cols)} metabolite features "
          f"(small {int((cls == 'small').sum())} / intermediate {int((cls == 'mid').sum())} / lipid {int((cls == 'lipid').sum())}; {n_inorg} inorganic / salt peaks removed), {len(ho)} baseline genes; {len(mod_targets)} modules")
    for level in ("full", "detrended"):
        prep = (lambda M_: C.zcols(M_)) if level == "full" else (lambda M_: C.zcols(C.detrend(M_, T)))
        Yl = prep(Ym); Tl = prep(Tg); Tj = C.zcols(Tl[jidx]); Ml = prep(np.column_stack(list(mod_targets.values()))) if mod_targets else None
        curves = {"metab": [], "gene": [], "gene_jit": [], "module": []}
        smo = {"metab": smooth_of(Yl, nbr), "gene": smooth_of(Tl, nbr), "gene_jit": smooth_of(Tj, nbr_j), "module": smooth_of(Ml, nbr) if Ml is not None else None}
        for s_ in SIG:
            K = Ks[s_]
            Xs = X0 if K is None else K @ X0; Xgs = Xg if K is None else K @ Xg
            Xs, Xgs = prep(Xs), prep(Xgs)
            Xs, Xgs = Xs[:, Xs.std(0) > 1e-9], Xgs[:, Xgs.std(0) > 1e-9]
            curves["metab"].append(r2cv(Xs, Yl, folds)); curves["gene"].append(r2cv(Xgs, Tl, folds))
            curves["gene_jit"].append(r2cv(C.zcols(Xgs[jok]), Tj, folds_j))
            if Ml is not None:
                curves["module"].append(r2cv(Xs, Ml, folds))
        for kind, nm_, extra in (("metab", names, cls), ("gene", gnames, gcls), ("gene_jit", gnames, np.char.add(gcls.astype(str), "_jit")),
                                 ("module", np.array(list(mod_targets)), np.array(["module"] * len(mod_targets)))):
            if not curves[kind]:
                continue
            Cv = np.array(curves[kind])                                   # sigma x feature
            for j in range(Cv.shape[1]):
                k_ = int(np.argmax(Cv[:, j]))
                frows.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=R, level=level, kind=kind, feature=nm_[j], cls=extra[j],
                                  mz=mz[j] if kind == "metab" else np.nan, annotation=ann[j] if kind == "metab" else "", R2_local=Cv[0, j],
                                  R2_best=Cv[k_, j], sigma_star=SIG[k_], context_gain=Cv[k_, j] - Cv[0, j], smooth=smo[kind][j]))
                for si, s_ in enumerate(SIG):
                    crows.append(dict(dataset=S["dataset"], sample=S["sample"], level=level, kind=kind, cls=extra[j], feature=nm_[j], sigma=s_, R2=Cv[si, j]))
        F_ = pd.DataFrame([r for r in frows if r["sample"] == S["sample"] and r["level"] == level])
        fm_, fg_, fj_ = F_[F_.kind == "metab"], F_[F_.kind == "gene"], F_[F_.kind == "gene_jit"]
        sm_, lp_ = fm_[fm_.cls == "small"].sigma_star, fm_[fm_.cls == "lipid"].sigma_star
        p_mg = mannwhitneyu(fm_.sigma_star, fg_.sigma_star, alternative="greater").pvalue if len(fg_) and len(fm_) else np.nan
        p_mj = mannwhitneyu(fm_.sigma_star, fj_.sigma_star, alternative="greater").pvalue if len(fj_) and len(fm_) else np.nan
        p_sl = mannwhitneyu(sm_, lp_, alternative="greater").pvalue if len(sm_) >= 3 and len(lp_) >= 3 else np.nan
        b_mj, pa_mj = adj_test(fm_, fj_); b_sl, pa_sl = adj_test(fm_[fm_.cls == "small"], fm_[fm_.cls == "lipid"])
        srows.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=R, level=level, same_section=bool(S.get("same_section", False)),
                          n_metab=len(fm_), n_small=len(sm_), n_lipid=len(lp_), n_gene=len(fg_),
                          jitter_um=a.jitter_um, n_jitter_bins=len(jok), smooth_metab=fm_.smooth.median(), smooth_gene_jit=fj_.smooth.median(),
                          beta_metab_vs_gene_jit_adj=b_mj, p_metab_gt_gene_jit_adj=pa_mj, beta_small_vs_lipid_adj=b_sl, p_small_gt_lipid_adj=pa_sl,
                          sigma_star_metab=fm_.sigma_star.median(), mean_sigma_metab=fm_.sigma_star.mean(), sigma_star_gene=fg_.sigma_star.median(),
                          sigma_star_gene_jit=fj_.sigma_star.median(), mean_sigma_gene_jit=fj_.sigma_star.mean(), p_metab_gt_gene=p_mg, p_metab_gt_gene_jit=p_mj,
                          sigma_star_small=sm_.median() if len(sm_) else np.nan, sigma_star_lipid=lp_.median() if len(lp_) else np.nan,
                          mean_sigma_small=sm_.mean() if len(sm_) else np.nan, mean_sigma_lipid=lp_.mean() if len(lp_) else np.nan, p_small_gt_lipid=p_sl,
                          gain_metab=fm_.context_gain.median(), gain_gene=fg_.context_gain.median(), gain_gene_jit=fj_.context_gain.median(), R2_local_metab=fm_.R2_local.median(),
                          R2_best_metab=fm_.R2_best.median(), sigma_star_module=F_[F_.kind == "module"].sigma_star.median() if (F_.kind == "module").any() else np.nan,
                          gain_module=F_[F_.kind == "module"].context_gain.median() if (F_.kind == "module").any() else np.nan))
        r = srows[-1]
        C.log(f"     [{level}] median sigma*: metabolites {r['sigma_star_metab']:.0f} µm vs genes {r['sigma_star_gene']:.0f} µm (p {r['p_metab_gt_gene']:.3g}) / "
              f"registration-matched genes {r['sigma_star_gene_jit']:.0f} µm (p {r['p_metab_gt_gene_jit']:.3g}); small {r['sigma_star_small']} vs lipid {r['sigma_star_lipid']} µm (p {r['p_small_gt_lipid']:.3g}); "
              f"context gain metabolites {r['gain_metab']:+.3f} / genes {r['gain_gene']:+.3f} / registration-matched genes {r['gain_gene_jit']:+.3f}; R² local {r['R2_local_metab']:+.3f} -> best {r['R2_best_metab']:+.3f}")
if not srows:
    raise SystemExit("no results")
FR, CR, SM = pd.DataFrame(frows), pd.DataFrame(crows), pd.DataFrame(srows)
SM["metab_gt_gene_jit"] = (SM.p_metab_gt_gene_jit < 0.05) & (SM.p_metab_gt_gene_jit_adj < 0.05)   # rank test and smoothness-adjusted test
for lv in ("full", "detrended"):
    SM.loc[SM.level == lv, "frac_samples_metab_gt_gene_jit"] = SM[SM.level == lv].metab_gt_gene_jit.mean()
SM["D43_context_beyond_registration"] = SM.frac_samples_metab_gt_gene_jit >= 0.7
z = norm.isf(SM.p_small_gt_lipid.clip(1e-12, 1 - 1e-12)); za = norm.isf(SM.p_small_gt_lipid_adj.clip(1e-12, 1 - 1e-12))
for lv in ("full", "detrended"):
    m = (SM.level == lv) & SM.p_small_gt_lipid.notna(); ma = (SM.level == lv) & SM.p_small_gt_lipid_adj.notna()
    SM.loc[SM.level == lv, "stouffer_p_small_gt_lipid"] = float(norm.sf(z[m].sum() / np.sqrt(m.sum()))) if m.sum() else np.nan
    SM.loc[SM.level == lv, "stouffer_p_small_gt_lipid_adj"] = float(norm.sf(za[ma].sum() / np.sqrt(ma.sum()))) if ma.sum() else np.nan
    SM.loc[SM.level == lv, "frac_samples_small_gt_lipid"] = float((SM[m].mean_sigma_small > SM[m].mean_sigma_lipid).mean()) if m.sum() else np.nan
SM["D44_small_longer_than_lipid"] = (SM.stouffer_p_small_gt_lipid < 0.05) & (SM.stouffer_p_small_gt_lipid_adj < 0.05) & (SM.frac_samples_small_gt_lipid >= 0.7)
SM["D45_context_gain"] = (SM.gain_metab >= 0.02) & (SM.gain_metab > SM.gain_gene_jit)
for df, f in ((FR, "m14_context_features.tsv"), (CR, "m14_context_curves.tsv"), (SM, "m14_context_summary.tsv")):
    df.to_csv(os.path.join(TAB, f), sep="\t", index=False)
C.log("summary:\n" + SM.round(3).to_string(index=False))
# figure: median R²(sigma)/max curve per sample and class
d = CR[CR.level == "full"].copy()
d["R2n"] = d.R2 / d.groupby(["sample", "kind", "feature"]).R2.transform(lambda v: max(v.max(), 1e-6))
smp = list(dict.fromkeys(d["sample"]))
fig, axs = plt.subplots(1, min(len(smp), 8) + 1, figsize=(2.6 * (min(len(smp), 8) + 1), 2.6), squeeze=False); axs = axs[0]
colr = {"gene": "#636363", "gene_jit": "#bdbdbd", "gene_pc": "#252525", "gene_pc_jit": "#969696", "small": "#de2d26", "mid": "#fd8d3c", "lipid": "#3182bd", "module": "#31a354"}
for ax_, s_ in zip(axs, smp[:8]):
    e = d[d["sample"] == s_]
    for c_, g in e.groupby("cls"):
        if c_ in colr and g.feature.nunique() >= 3:
            m_ = g.groupby("sigma").R2n.median(); ax_.plot(m_.index, m_.values, "o-", ms=2.5, c=colr[c_], label=f"{c_} (n={g.feature.nunique()})")
    ax_.set_xscale("symlog", linthresh=100); ax_.set_title(str(s_)[:18], fontsize=6.5); ax_.set_xlabel("kernel σ (µm)"); ax_.legend(fontsize=4.5, frameon=False)
axs[0].set_ylabel("R²(σ) / max R² (median)")
f_ = FR[(FR.level == "full")]
groups = [("gene", f_[f_.kind == "gene"].sigma_star), ("gene_jit", f_[f_.kind == "gene_jit"].sigma_star), ("small", f_[(f_.kind == "metab") & (f_.cls == "small")].sigma_star),
          ("mid", f_[(f_.kind == "metab") & (f_.cls == "mid")].sigma_star), ("lipid", f_[(f_.kind == "metab") & (f_.cls == "lipid")].sigma_star),
          ("module", f_[f_.kind == "module"].sigma_star)]
groups = [(k, v) for k, v in groups if len(v)]
axs[-1].boxplot([v for _, v in groups], showfliers=False); axs[-1].set_xticks(np.arange(1, len(groups) + 1)); axs[-1].set_xticklabels([k for k, _ in groups], rotation=45); axs[-1].set_ylabel("σ* (µm), all samples")
axs[-1].set_title("context length by class", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m14_context_length.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
