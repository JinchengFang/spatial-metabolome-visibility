#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="ccRCC,GBM,SMA,SMAx")
ap.add_argument("--res", type=int, default=200); ap.add_argument("--sigma-ctx", type=float, default=400.0); ap.add_argument("--max-feat", type=int, default=100)
ap.add_argument("--n-null", type=int, default=49); ap.add_argument("--n-gene", type=int, default=100); ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--min-det", type=float, default=0.10)
ap.add_argument("--cluster-r", type=float, default=0.9); ap.add_argument("--min-cover", type=float, default=0.5); ap.add_argument("--min-sb", type=float, default=0.3)
ap.add_argument("--ccrcc-dir", default=None); ap.add_argument("--ion-set", default="strict"); ap.add_argument("--include-uncertain", action="store_true")
ap.add_argument("--no-shift", action="store_true"); ap.add_argument("--keep-inorganic", action="store_true"); ap.add_argument("--step", default="Step12")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
cfg = L.Config(P, ion_set=a.ion_set, ccrcc_dir=a.ccrcc_dir, include_uncertain=a.include_uncertain, min_px_half=1, min_bins=80)
DS = [d.strip() for d in a.datasets.split(",") if d.strip()]; rng = np.random.default_rng(0)
SHIFT = {}
for st_ in ("Step8", "Step10", "Step11"):
    fs = os.path.join(P["P2"], "05_tables", st_, "m11_registration_summary.tsv")
    if os.path.exists(fs) and not a.no_shift:
        rs = pd.read_csv(fs, sep="\t"); rs = rs[(rs.res_um == rs.res_um.min()) & (rs.cv_gain > 0.02)]
        SHIFT.update({r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples()})
HCOLS = [c for g in C.HISTO_GROUPS.values() for c in g]


def kernel(xy, s):
    tree = cKDTree(xy); pr = tree.sparse_distance_matrix(tree, 3 * s, output_type="coo_matrix")
    K = sp.csr_matrix((np.exp(-0.5 * (pr.data / s) ** 2), (pr.row, pr.col)), shape=(len(xy), len(xy))) + sp.eye(len(xy))
    return sp.diags(1 / np.asarray(K.sum(1)).ravel()) @ K


def r2(X, Y, folds):
    X = X[:, X.std(0) > 1e-9]
    return C.r2_columns(Y, C.cv_ridge_gcv(X, Y, folds))


def budget(rT, rTC, rTCH, SB, rel):
    """Additive variance budget: R²/SB of every reliable feature clipped to [0, 1] and averaged; each part is the difference of successive models; the four parts sum to 1."""
    if not rel.any():
        return dict(budget_local=np.nan, budget_context=np.nan, budget_histo=np.nan, budget_unexplained=np.nan)
    f_ = lambda r_: float(np.mean(np.clip(r_[rel] / SB[rel], 0, 1)))
    a1, a2, a3 = f_(rT), f_(rTC), f_(rTCH); a2, a3 = max(a2, a1), max(a3, max(a2, a1))
    return dict(budget_local=a1, budget_context=a2 - a1, budget_histo=a3 - a2, budget_unexplained=1 - a3)


frows, srows, grows, cons = [], [], [], {}
for S in L.load_all(cfg, DS):
    if S.get("he") is None:
        C.log(f"  {S['dataset']} {S['sample']}: no H&E available; skipped"); continue
    R = a.res; G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    sh = SHIFT.get(S["sample"], (0.0, 0.0))
    Y0, cnt = L.pixel_means(S, G, None, sh); YA, _ = L.pixel_means(S, G, S["px_half"] == 0, sh); YB, _ = L.pixel_means(S, G, S["px_half"] == 1, sh)
    im = C.load_he(S["he"])
    cpx = G["xy"] * S["he"]["scale"] + np.asarray(S["he"]["offset"]); Hdf, cover = C.histo_features(im, cpx, R / 2 * S["he"]["scale"])
    ok = (cnt >= 1) & (cover >= a.min_cover) & Hdf[HCOLS].notna().all(1).values
    if ok.sum() < 80:
        C.log(f"  {S['dataset']} {S['sample']}: only {ok.sum()} bins covered by H&E (< 80); skipped (check image-coordinate correspondence)"); continue
    xy = G["xy"][ok]
    det = (np.expm1(Y0[ok]) > 0).mean(0); okf = np.where(det >= a.min_det)[0]
    if S["mz"] is not None and not a.keep_inorganic:
        okf = okf[C.organic_mass_defect_ok(np.asarray(S["mz"], float)[okf])]
    reps, _, _, _ = C.select_structured_features(C.zcols(Y0[ok][:, okf]), xy, a.max_feat, a.cluster_r, seed=R)
    if len(reps) < 5:
        continue
    cols = okf[reps]; names = np.asarray(S["names"]).astype(str)[cols]
    mz = np.asarray(S["mz"], float)[cols] if S["mz"] is not None else np.full(len(cols), np.nan)
    CB = G["CB"][np.where(ok)[0]]; Bk, LG, _ = L.tx_blocks(S, CB, G["ncell"][ok], None if G["comp_fixed"] is None else G["comp_fixed"][ok])
    X0 = np.column_stack([np.asarray(v, float) for v in Bk.values()]); X0 = X0[:, X0.std(0) > 1e-9]
    H0 = Hdf[HCOLS].values[ok].astype(float); K = kernel(xy, a.sigma_ctx); folds = C.spatial_folds(xy, a.block_um, seed=R); T2 = C.poly2(xy)
    # gene control: if H&E were only a cleaner readout of cell composition, its increment for held-out genes would match that for metabolites
    gv = LG.var(0); cand = np.argsort(-gv)[:2000]; cand = cand[gv[cand] > 0]
    ho = np.sort(rng.choice(cand, min(a.n_gene, len(cand) // 3), replace=False)); keepg = np.setdiff1d(np.arange(CB.shape[1]), ho)
    CA_, CH_ = C.thin_counts(CB, rng)
    Bg, _, _ = C.tx_blocks_from_bins(CA_[:, keepg], np.asarray(S["genes"])[keepg], sig=S["sig"], programs_fallback=S["programs"], state_sets=S["state_sets"],
                                     state_min_genes=S["state_min"], comp_fixed=None if G["comp_fixed"] is None else G["comp_fixed"][ok])
    Xg = np.column_stack([np.asarray(v, float) for v in Bg.values()]); Xg = Xg[:, Xg.std(0) > 1e-9]
    totH = np.asarray(CH_.sum(1)).ravel(); Tg = np.log1p((sp.diags(1e4 / np.maximum(totH, 1)) @ CH_[:, ho]).toarray())
    Uh, Sh, _ = np.linalg.svd(C.zcols(Tg), full_matrices=False); Tg = np.column_stack([Tg, Uh[:, :10] * Sh[:10]])
    # rotation null: H of the nearest bin after rotation about the tissue centre
    ctr = xy.mean(0); tree = cKDTree(xy); NN = []
    for th in rng.uniform(np.deg2rad(45), np.deg2rad(315), a.n_null):
        c_, s_ = np.cos(th), np.sin(th); q = (xy - ctr) @ np.array([[c_, -s_], [s_, c_]]).T + ctr; NN.append(tree.query(q)[1])
    C.log(f"  {S['dataset']} {S['sample']} @ {R} µm{' (shift ' + str(sh) + ')' if any(sh) else ''}: {ok.sum()} bins (H&E-covered), {len(cols)} metabolite features")
    for level in ("full", "detrended"):
        prep = (lambda M_: C.zcols(M_)) if level == "full" else (lambda M_: C.zcols(C.detrend(M_, T2)))
        Y = prep(Y0[ok][:, cols]); _, SB = C.split_half_reliability(prep(YA[ok][:, cols]), prep(YB[ok][:, cols]))
        T = prep(X0); TC = np.column_stack([T, prep(K @ X0)]); HC = np.column_stack([prep(H0), prep(K @ H0)])
        r_T, r_TC, r_H = r2(T, Y, folds), r2(TC, Y, folds), r2(HC, Y, folds); r_TCH = r2(np.column_stack([TC, HC]), Y, folds)
        dH = r_TCH - r_TC
        Tgl = prep(Tg); TCg = np.column_stack([prep(Xg), prep(K @ Xg)])
        dHg = r2(np.column_stack([TCg, HC]), Tgl, folds) - r2(TCg, Tgl, folds); dHg_pc = dHg[-10:]
        orth = SB - r_TC; use = (SB >= a.min_sb) & (orth > 0.1)          # reliable features with > 0.1 left after transcriptome + context
        fo = lambda d_: float(np.median(np.clip(d_[use] / orth[use], -1, 1))) if use.sum() >= 5 else np.nan
        rE = np.where(use, np.clip(dH / np.where(orth > 0, orth, np.nan), -1, 1), np.nan)
        null_d, null_e = [], []
        for nn in NN:
            dn = r2(np.column_stack([TC, HC[nn]]), Y, folds) - r_TC
            null_d.append(np.median(dn)); null_e.append(fo(dn))
        gi = {g: [HCOLS.index(c) for c in cs] for g, cs in C.HISTO_GROUPS.items()}
        gcon = {}
        for g, idx in gi.items():
            keep_c = [j for j in range(len(HCOLS)) if j not in idx]; HCg = np.column_stack([HC[:, keep_c], HC[:, [len(HCOLS) + j for j in keep_c]]])
            gcon[g] = float(np.median(r_TCH - r2(np.column_stack([TC, HCg]), Y, folds)))
            grows.append(dict(dataset=S["dataset"], sample=S["sample"], level=level, group=g, median_drop_in_R2=gcon[g]))
        rel = SB >= a.min_sb
        for j in range(len(cols)):
            sb = SB[j] if np.isfinite(SB[j]) else np.nan
            frows.append(dict(dataset=S["dataset"], sample=S["sample"], level=level, feature=names[j], mz=mz[j], SB=sb, R2_T=r_T[j], R2_TC=r_TC[j],
                              R2_TCH=r_TCH[j], R2_H=r_H[j], delta_H=dH[j], frac_orth_explained_by_H=rE[j],
                              frac_local=r_T[j] / sb if sb == sb and sb > 0 else np.nan, frac_context=(r_TC[j] - r_T[j]) / sb if sb == sb and sb > 0 else np.nan,
                              frac_histo=dH[j] / sb if sb == sb and sb > 0 else np.nan))
        srows.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=R, level=level, design=("same-section" if S["dataset"] == "SMA-mouse" else
                          "SMA-cross-control" if S["dataset"] == "SMA-cross" else "adjacent-section"),
                          n_bins=int(ok.sum()), n_feat=len(cols), n_reliable=int(rel.sum()), SB_median=float(np.nanmedian(SB)),
                          R2_T=float(np.median(r_T)), R2_TC=float(np.median(r_TC)), R2_TCH=float(np.median(r_TCH)), R2_H_alone=float(np.median(r_H)),
                          delta_H=float(np.median(dH)), delta_H_genes=float(np.median(dHg[:-10])), delta_H_gene_pcs=float(np.median(dHg_pc)),
                          p_metab_gt_gene=float(__import__("scipy.stats", fromlist=["mannwhitneyu"]).mannwhitneyu(dH, dHg, alternative="greater").pvalue),
                          delta_H_null95=float(np.percentile(null_d, 95)), p_delta_H=float((np.sum(np.array(null_d) >= np.median(dH)) + 1) / (len(null_d) + 1)),
                          n_orth_features=int(use.sum()), frac_orth_explained=fo(dH), frac_orth_null95=float(np.nanpercentile(null_e, 95)) if np.isfinite(null_e).any() else np.nan,
                          p_orth=float((np.sum(np.array(null_e) >= fo(dH)) + 1) / (len(null_e) + 1)) if np.isfinite(fo(dH)) else np.nan,
                          **budget(r_T, r_TC, r_TCH, SB, rel),
                          top_group=max(gcon, key=gcon.get), **{f"group_{g}": v for g, v in gcon.items()}))
        r_ = srows[-1]
        C.log(f"     [{level}] R² transcriptome {r_['R2_T']:+.3f} -> +context {r_['R2_TC']:+.3f} -> +histology {r_['R2_TCH']:+.3f} (dH {r_['delta_H']:+.3f}, null 95% {r_['delta_H_null95']:+.3f}, p {r_['p_delta_H']:.3f}; held-out genes dH {r_['delta_H_genes']:+.3f} / gene PCs {r_['delta_H_gene_pcs']:+.3f}); "
              f"orthogonal reliable variation explained by histology {r_['frac_orth_explained']:+.3f} (null 95% {r_['frac_orth_null95']:+.3f}, {r_['n_orth_features']} features); histology alone {r_['R2_H_alone']:+.3f}; top group {r_['top_group']}")
        if level == "full":
            cons[S["sample"]] = (S["dataset"], pd.Series(dH, index=names))
    # figure: H&E + bin features + metabolite with the largest histology increment
    try:
        j = int(np.argmax(dH)); fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
        axs[0].imshow(im); axs[0].scatter(cpx[ok, 0], cpx[ok, 1], s=1, c="k", alpha=.3); axs[0].set_title(f"{S['sample']} H&E", fontsize=7)
        for ax_, v_, t_ in ((axs[1], H0[:, HCOLS.index("red_frac")], "red (RBC / haemorrhage) fraction"), (axs[2], H0[:, HCOLS.index("white_frac")], "white (lumen / void) fraction"),
                            (axs[3], Y[:, j], f"{names[j]}  ΔH {dH[j]:+.2f}")):
            ax_.scatter(xy[:, 0], xy[:, 1], c=v_, s=6, cmap="magma", marker="s"); ax_.invert_yaxis(); ax_.set_aspect("equal"); ax_.set_title(t_, fontsize=7); ax_.set_xticks([]); ax_.set_yticks([])
        axs[0].set_xticks([]); axs[0].set_yticks([]); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m16_histology_{S['sample']}.png"), dpi=150); plt.close(fig)
    except Exception as e:
        C.log(f"     (figure failed: {e})")
if not srows:
    raise SystemExit("no sample with usable H&E (ccRCC: uns['spatial'] of the h5ad; GBM: uns['hires_image'] of ravi_paired/*_visium.h5ad; SMA: ST/*tissue_hires_image.png)")
FR, SM, GR = pd.DataFrame(frows), pd.DataFrame(srows), pd.DataFrame(grows)
SM["D49_histology_increment"] = (SM.delta_H >= 0.02) & (SM.delta_H > SM.delta_H_null95) & (SM.p_metab_gt_gene < 0.05)
SM["D50_histology_explains_orthogonal"] = (SM.frac_orth_explained >= 0.10) & (SM.frac_orth_explained > SM.frac_orth_null95)
# D51: consistency of dH vectors between ccRCC patients
crow = []
for ds in sorted({v[0] for v in cons.values()}):
    smp = [k for k, v in cons.items() if v[0] == ds]
    for i in range(len(smp)):
        for k in range(i + 1, len(smp)):
            u, w = cons[smp[i]][1], cons[smp[k]][1]; com = u.index.intersection(w.index)
            com = com[~com.duplicated()]
            if len(com) < 15:
                continue
            uu, ww = u[~u.index.duplicated()].reindex(com).values, w[~w.index.duplicated()].reindex(com).values
            rho = spearmanr(uu, ww).correlation; nul = [spearmanr(uu, rng.permutation(ww)).correlation for _ in range(999)]
            crow.append(dict(dataset=ds, sample_a=smp[i], sample_b=smp[k], n_common=len(com), rho=rho, p=(np.sum(np.array(nul) >= rho) + 1) / 1000))
CS = pd.DataFrame(crow)
for df, f in ((FR, "m16_histology_features.tsv"), (SM, "m16_histology_summary.tsv"), (GR, "m16_group_contrib.tsv"), (CS, "m16_conservation.tsv")):
    df.to_csv(os.path.join(TAB, f), sep="\t", index=False)
C.log("summary:\n" + SM.drop(columns=[c for c in SM if c.startswith("group_")]).round(3).to_string(index=False))
if len(CS):
    C.log("consistency of dH across samples:\n" + CS.round(3).to_string(index=False))
# variance-budget figure
s = SM[SM.level == "full"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(s) + 2), 3.4)); x = np.arange(len(s)); bot = np.zeros(len(s))
for c_, col, lab in (("budget_local", "#3182bd", "local transcriptome"), ("budget_context", "#9ecae1", "+ transcriptomic context (σ 400 µm)"),
                     ("budget_histo", "#fd8d3c", "+ H&E histology"), ("budget_unexplained", "#d9d9d9", "unexplained (reliable)")):
    v = s[c_].fillna(0).values; ax.bar(x, v, bottom=bot, color=col, label=lab, width=.75); bot = bot + v
ax.set_xticks(x); ax.set_xticklabels([f"{r.dataset}\n{str(r.sample)[:14]}" for r in s.itertuples()], rotation=60, fontsize=5)
ax.set_ylabel("share of reliable metabolite variance\n(mean over features, R²/ceiling)"); ax.legend(fontsize=5, frameon=False, ncol=2); ax.set_ylim(0, 1.05)
ax.set_title("metabolic variance budget (200 µm, full)", fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "m16_variance_budget.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
