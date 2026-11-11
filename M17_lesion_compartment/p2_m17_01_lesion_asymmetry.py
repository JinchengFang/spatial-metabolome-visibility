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
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="SMA"); ap.add_argument("--res", type=int, default=200)
ap.add_argument("--top", type=int, default=30); ap.add_argument("--sigma-ctx", type=float, default=400.0); ap.add_argument("--max-feat", type=int, default=300)
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--keep-inorganic", action="store_true")
ap.add_argument("--no-shift", action="store_true"); ap.add_argument("--step", default="Step12")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
cfg = L.Config(P, min_px_half=1, min_bins=80); rng = np.random.default_rng(0)
SHIFT = {}
for st_ in ("Step10", "Step11"):
    fs = os.path.join(P["P2"], "05_tables", st_, "m11_registration_summary.tsv")
    if os.path.exists(fs) and not a.no_shift:
        rs = pd.read_csv(fs, sep="\t"); rs = rs[(rs.res_um == rs.res_um.min()) & (rs.cv_gain > 0.02)]
        SHIFT.update({r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples()})


def kernel(xy, s):
    tree = cKDTree(xy); pr = tree.sparse_distance_matrix(tree, 3 * s, output_type="coo_matrix")
    K = sp.csr_matrix((np.exp(-0.5 * (pr.data / s) ** 2), (pr.row, pr.col)), shape=(len(xy), len(xy))) + sp.eye(len(xy))
    return sp.diags(1 / np.asarray(K.sum(1)).ravel()) @ K


def reflect(xy, th, d, c):
    """Reflection about the line through c + d n with direction (cos th, sin th) (n is the normal)."""
    u = np.array([np.cos(th), np.sin(th)]); n = np.array([-u[1], u[0]]); q = xy - c - d * n
    return q - 2 * np.outer(q @ n, n) + c + d * n


def mirror_axis(xy, PC):
    """Left-right symmetry of Visium gene principal components. Returns (th, d, score); score = mean PC correlation over mirrored spot pairs (< 75 µm)."""
    c = xy.mean(0); tree = cKDTree(xy); best = (-2, 0.0, 0.0)
    def sc(th, d):
        q = reflect(xy, th, d, c); dd, j = tree.query(q); ok = dd < 75
        if ok.sum() < 0.3 * len(xy):
            return -2
        return float(np.mean([np.corrcoef(PC[ok, k], PC[j[ok], k])[0, 1] for k in range(PC.shape[1])]))
    for th in np.deg2rad(np.arange(0, 180, 2.0)):
        for d in np.arange(-1500, 1501, 100):
            s_ = sc(th, d)
            if s_ > best[0]:
                best = (s_, th, d)
    s0, th0, d0 = best
    for th in th0 + np.deg2rad(np.arange(-2, 2.01, 0.5)):
        for d in d0 + np.arange(-100, 101, 25):
            s_ = sc(th, d)
            if s_ > best[0]:
                best = (s_, th, d)
    return best[1], best[2], best[0]


def asym(V, I, J):
    D = V[I] - V[J]; return D.mean(0), D.mean(0) / np.maximum(D.std(0), 1e-9)


frows, srows, grows, vecs = [], [], [], {}
for S in L.load_all(cfg, [d.strip() for d in a.datasets.split(",")]):
    lab = (S.get("tx_labels") or {}).get("lesion")
    if lab is None:
        C.log(f"  {S['dataset']} {S['sample']}: no lesion annotation; skipped"); continue
    R = a.res; G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    sh = SHIFT.get(S["sample"], (0.0, 0.0))
    Y0, cnt = L.pixel_means(S, G, None, sh); YA, _ = L.pixel_means(S, G, S["px_half"] == 0, sh); YB, _ = L.pixel_means(S, G, S["px_half"] == 1, sh)
    # lesion label of each bin = majority of its spots
    kt = np.floor((S["tx_xy"] - G["o"]) / R).astype(np.int64); it = G["gi"].reindex(kt[:, 1] * 100000 + kt[:, 0]).fillna(-1).astype(int).values
    lab_s = pd.Series(np.asarray(lab).astype(str)).str.lower()
    nint = np.bincount(it[(it >= 0) & lab_s.str.startswith("intact").values], minlength=len(G["keys"]))
    nles = np.bincount(it[(it >= 0) & lab_s.str.startswith("lesion").values], minlength=len(G["keys"]))
    blab = np.where(nint > nles, "intact", np.where(nles > nint, "lesioned", "NA"))
    ok = cnt >= 1; xy = G["xy"][ok]; blab = blab[ok]
    # mirror axis (Visium spots, gene principal components)
    CTs = S["counts"]; tot = np.asarray(CTs.sum(1)).ravel(); LGs = (sp.diags(1e4 / np.maximum(tot, 1)) @ CTs).tocsr(); LGs.data = np.log1p(LGs.data)
    v = np.asarray(LGs.power(2).mean(0)).ravel() - np.asarray(LGs.mean(0)).ravel() ** 2; hv = np.argsort(-v)[:2000]
    Zs = C.zcols(LGs[:, hv].toarray()); U, Sv, _ = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); PCs = U[:, :10] * Sv[:10]
    th, d, msc = mirror_axis(S["tx_xy"], PCs)
    q = reflect(xy, th, d, S["tx_xy"].mean(0)); dd, jj = cKDTree(xy).query(q)
    I = np.where((blab == "intact") & (dd < 0.75 * R) & (blab[jj] == "lesioned"))[0]; J = jj[I]
    C.log(f"  {S['dataset']} {S['sample']}: mirror axis {np.rad2deg(th):.1f} deg, offset {d:+.0f} µm (mirrored PC correlation {msc:.2f}); {len(I)} intact-lesioned bin pairs")
    if len(I) < 30:
        C.log("     fewer than 30 mirrored bin pairs; skipped"); continue
    # metabolite features
    det = (np.expm1(Y0[ok]) > 0).mean(0); okf = np.where(det >= a.min_det)[0]
    if S["mz"] is not None and not a.keep_inorganic:
        okf = okf[C.organic_mass_defect_ok(np.asarray(S["mz"], float)[okf])]
    reps, _, _, _ = C.select_structured_features(C.zcols(Y0[ok][:, okf]), xy, a.max_feat, 0.9, seed=R)
    cols = okf[reps]; names = np.asarray(S["names"]).astype(str)[cols]; mz = np.asarray(S["mz"], float)[cols]
    Y = C.zcols(Y0[ok][:, cols]); ZA, ZB = C.zcols(YA[ok][:, cols]), C.zcols(YB[ok][:, cols])
    _, dA = asym(ZA, I, J); DB, dB = asym(ZB, I, J); DY, dY = asym(Y, I, J)
    sel = np.argsort(-np.abs(dA))[: a.top]; rep_ = (np.sign(dB[sel]) == np.sign(dA[sel])) & (np.abs(dB[sel]) >= 0.3)
    # cross-fitted prediction and residuals of the transcriptome (+ context)
    CB = G["CB"][np.where(ok)[0]]; Bk, LG, _ = L.tx_blocks(S, CB, G["ncell"][ok], None)
    X0 = np.column_stack([np.asarray(v_, float) for v_ in Bk.values()]); X0 = X0[:, X0.std(0) > 1e-9]; K = kernel(xy, a.sigma_ctx)
    TC = C.zcols(np.column_stack([X0, K @ X0])); TC = TC[:, TC.std(0) > 1e-9]; folds = C.spatial_folds(xy, a.block_um, seed=R)
    Yh = C.cv_ridge_gcv(TC, Y, folds); E = Y - Yh; DE, _ = asym(E, I, J); Dh, _ = asym(Yh, I, J)
    r2 = C.r2_columns(Y, Yh); fracE = DE / np.where(np.abs(DY) > 1e-9, DY, np.nan)
    for k, j in enumerate(sel):
        frows.append(dict(dataset=S["dataset"], sample=S["sample"], feature=names[j], mz=mz[j], d_A=dA[j], d_B=dB[j], replicated=bool(rep_[k]),
                          delta_Y=DY[j], delta_pred=Dh[j], delta_resid=DE[j], frac_asym_orthogonal=fracE[j], R2_TC=r2[j],
                          higher_in="intact" if DY[j] > 0 else "lesioned"))
    vecs[S["sample"]] = (S.get("matrix", ""), pd.Series(dY, index=names))
    # positive control: genes (binomially thinned halves)
    CA_, CH_ = C.thin_counts(CB, rng); gsel = np.argsort(-LG.var(0))[:2000]
    nrm = lambda M_: np.log1p((sp.diags(1e4 / np.maximum(np.asarray(M_.sum(1)).ravel(), 1)) @ M_[:, gsel]).toarray())
    GA, GB = C.zcols(nrm(CA_)), C.zcols(nrm(CH_)); _, gdA = asym(GA, I, J); _, gdB = asym(GB, I, J)
    gtop = np.argsort(-np.abs(np.nan_to_num(gdA)))[: a.top]; grep_ = (np.sign(gdB[gtop]) == np.sign(gdA[gtop])) & (np.abs(gdB[gtop]) >= 0.3)
    gn = np.asarray(S["genes"])[gsel]
    for k, j in enumerate(gtop):
        grows.append(dict(dataset=S["dataset"], sample=S["sample"], gene=gn[j], d_A=gdA[j], d_B=gdB[j], replicated=bool(grep_[k]),
                          higher_in="intact" if gdA[j] > 0 else "lesioned"))
    rf = fracE[sel][rep_]
    srows.append(dict(dataset=S["dataset"], sample=S["sample"], matrix=S.get("matrix", ""), mirror_score=msc, n_pairs=len(I), n_feat=len(cols),
                      n_top=len(sel), frac_replicated=float(rep_.mean()), n_replicated=int(rep_.sum()), median_abs_dB_replicated=float(np.median(np.abs(dB[sel][rep_]))) if rep_.any() else np.nan,
                      median_frac_asym_orthogonal=float(np.nanmedian(rf)) if rep_.any() else np.nan, median_R2_TC_replicated=float(np.median(r2[sel][rep_])) if rep_.any() else np.nan,
                      median_R2_TC_all=float(np.median(r2)), n_genes_replicated=int(grep_.sum()), frac_genes_replicated=float(grep_.mean())))
    r_ = srows[-1]
    C.log(f"     metabolites: {r_['n_replicated']} of the top {len(sel)} asymmetric features replicated ({r_['frac_replicated']:.0%}); median transcriptome-orthogonal fraction of the asymmetry of replicated features {r_['median_frac_asym_orthogonal']:.2f} "
          f"(R²_TC {r_['median_R2_TC_replicated']:+.2f}, all features {r_['median_R2_TC_all']:+.2f}); genes: {r_['n_genes_replicated']} / {len(gtop)} replicated")
    # figure: mirror axis + most asymmetric feature (measured / transcriptome prediction / residual)
    try:
        j = sel[np.argmax(np.abs(dA[sel]) * rep_)] if rep_.any() else sel[0]
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
        axs[0].scatter(xy[:, 0], xy[:, 1], c=np.where(blab == "intact", 0, np.where(blab == "lesioned", 1, .5)), cmap="coolwarm", s=6, marker="s")
        c0 = S["tx_xy"].mean(0); u = np.array([np.cos(th), np.sin(th)]); n_ = np.array([-u[1], u[0]]); p0 = c0 + d * n_
        axs[0].plot([p0[0] - 4000 * u[0], p0[0] + 4000 * u[0]], [p0[1] - 4000 * u[1], p0[1] + 4000 * u[1]], "k--", lw=.8)
        axs[0].set_title(f"intact (blue) / lesioned (red); mirror r {msc:.2f}", fontsize=7)
        for ax_, v_, t_ in ((axs[1], Y[:, j], f"measured {names[j]}"), (axs[2], Yh[:, j], f"transcriptome-predicted (R² {r2[j]:.2f})"),
                            (axs[3], E[:, j], f"orthogonal residual (Δ_E/Δ_Y {fracE[j]:.2f})")):
            ax_.scatter(xy[:, 0], xy[:, 1], c=v_, cmap="viridis", s=6, marker="s"); ax_.set_title(t_, fontsize=7)
        for ax_ in axs:
            ax_.invert_yaxis(); ax_.set_aspect("equal"); ax_.set_xticks([]); ax_.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m17_mirror_{S['sample']}.png"), dpi=150); plt.close(fig)
    except Exception as e:
        C.log(f"     (figure failed: {e})")
if not srows:
    raise SystemExit("no SMA section with lesion annotation")
FR, SM, GR = pd.DataFrame(frows), pd.DataFrame(srows), pd.DataFrame(grows)
# consistency of asymmetry vectors between sections of the same matrix
crow = []
ks = list(vecs)
for i in range(len(ks)):
    for k in range(i + 1, len(ks)):
        if vecs[ks[i]][0] != vecs[ks[k]][0]:
            continue
        u_, w_ = vecs[ks[i]][1], vecs[ks[k]][1]; com = u_.index.intersection(w_.index).unique()
        if len(com) >= 20:
            uu, ww = u_[~u_.index.duplicated()].reindex(com).values, w_[~w_.index.duplicated()].reindex(com).values
            rho = spearmanr(uu, ww).correlation; nul = [spearmanr(uu, rng.permutation(ww)).correlation for _ in range(999)]
            crow.append(dict(sample_a=ks[i], sample_b=ks[k], matrix=vecs[ks[i]][0], n_common=len(com), rho=rho, p=(np.sum(np.array(nul) >= rho) + 1) / 1000))
CS = pd.DataFrame(crow)
cons_ok = set()
for r in CS.itertuples():
    if r.rho >= 0.3 and r.p < 0.05:
        cons_ok |= {r.sample_a, r.sample_b}
SM["consistent_across_sections"] = SM["sample"].isin(cons_ok)
SM["D52_reproducible_asymmetry"] = (SM.frac_replicated >= 0.5) | SM.consistent_across_sections
SM["D53_asymmetry_orthogonal"] = SM.median_frac_asym_orthogonal >= 0.5
SM["D53g_gene_positive_control"] = SM.n_genes_replicated >= 10
for df, f in ((FR, "m17_asymmetry_features.tsv"), (SM, "m17_asymmetry_summary.tsv"), (GR, "m17_asymmetry_genes.tsv"), (CS, "m17_asymmetry_consistency.tsv")):
    df.to_csv(os.path.join(TAB, f), sep="\t", index=False)
C.log("summary:\n" + SM.round(3).to_string(index=False) + ("\nconsistency across sections:\n" + CS.round(3).to_string(index=False) if len(CS) else ""))
C.log("strongest replicated metabolite asymmetries:\n" + FR[FR.replicated].sort_values("d_B", key=np.abs, ascending=False).head(20).round(3).to_string(index=False))
C.log("strongest replicated gene asymmetries:\n" + GR[GR.replicated].sort_values("d_B", key=np.abs, ascending=False).head(20).round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(4, 3.3))
f_ = FR[FR.replicated]
ax.scatter(f_.delta_Y, f_.delta_pred, s=8, c="#de2d26"); lim = np.nanmax(np.abs(np.r_[f_.delta_Y, f_.delta_pred, 0.1])) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", lw=.6); ax.axhline(0, c="grey", lw=.4); ax.axvline(0, c="grey", lw=.4)
ax.set_xlabel("measured lesion asymmetry Δ_Y (z units)"); ax.set_ylabel("transcriptome-predicted asymmetry"); ax.set_title("lesion-induced metabolic asymmetry", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m17_asymmetry.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
