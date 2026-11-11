#!/usr/bin/env python3

import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200); ap.add_argument("--sigma-ctx", type=float, default=400.0)
ap.add_argument("--max-feat", type=int, default=200); ap.add_argument("--n-rand", type=int, default=999); ap.add_argument("--min-det", type=float, default=0.05)
ap.add_argument("--matrix", default="FMP10"); ap.add_argument("--ppm", type=float, default=5.0); ap.add_argument("--min-sb", type=float, default=0.2)
ap.add_argument("--step", default="Step13")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
cfg = L.Config(P, min_px_half=1, min_bins=60); rng = np.random.default_rng(0)


def kernel(xy, s):
    tree = cKDTree(xy); pr = tree.sparse_distance_matrix(tree, 3 * s, output_type="coo_matrix")
    K = sp.csr_matrix((np.exp(-0.5 * (pr.data / s) ** 2), (pr.row, pr.col)), shape=(len(xy), len(xy))) + sp.eye(len(xy))
    return sp.diags(1 / np.asarray(K.sum(1)).ravel()) @ K


ANN, rows, maps = None, [], {}
for S in L.load_all(cfg, ["SMA"]):
    if a.matrix.lower() not in str(S.get("matrix", "")).lower():
        continue
    sec = S.get("msi_section", S["sample"]); area = S.get("brain_area", "")
    import anndata as ad
    V = ad.read_h5ad(os.path.join(P["PROC"], "sma", f"{sec}_paired.h5ad"), backed="r").var
    if ANN is None:
        ANN = C.annotate_fmp10(V.mz.values, V.win_lo.values, V.win_hi.values, ppm=a.ppm)
        ANN["peak_name"] = np.asarray(S["names"]).astype(str)[ANN.peak_index.values]
        ANN.to_csv(os.path.join(TAB, "m18_fmp10_annotation.tsv"), sep="\t", index=False)
        C.log("FMP-10 annotation (shared peak list):\n" + ANN.round(4).to_string(index=False))
    R = a.res; G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    Y0, cnt = L.pixel_means(S, G, None); YA, _ = L.pixel_means(S, G, S["px_half"] == 0); YB, _ = L.pixel_means(S, G, S["px_half"] == 1)
    ok = cnt >= 1; xy = G["xy"][ok]; Yk, YAk, YBk = Y0[ok], YA[ok], YB[ok]
    det = (np.expm1(Yk) > 0).mean(0); okf = np.where(det >= a.min_det)[0]
    okf = okf[C.organic_mass_defect_ok(np.asarray(S["mz"], float)[okf])]
    reps, _, _, _ = C.select_structured_features(C.zcols(Yk[:, okf]), xy, a.max_feat, 0.95, seed=R); cols = okf[reps]
    # molecule targets
    tg, tgA, tgB, tginfo = [], [], [], []
    for mol, g in ANN.groupby("molecule"):
        idx = [i for i in g.peak_index.unique() if det[i] >= a.min_det]
        if not idx:
            continue
        Z, ZA, ZB = C.zcols(Yk[:, idx]), C.zcols(YAk[:, idx]), C.zcols(YBk[:, idx])
        cons = float(np.nanmean(np.corrcoef(Z.T)[np.triu_indices(len(idx), 1)])) if len(idx) > 1 else np.nan
        tg.append(np.nan_to_num(Z).mean(1)); tgA.append(np.nan_to_num(ZA).mean(1)); tgB.append(np.nan_to_num(ZB).mean(1))
        tginfo.append(dict(molecule=mol, n_ions=len(idx), ions="; ".join(f"{f}@{m:.4f}" for f, m in zip(g.form, g.observed_mz) if True)[:300], ion_consistency=cons,
                           shared_peak=bool((g.n_molecules_on_peak > 1).any())))
    if not tg:
        continue
    Tm, TmA, TmB = np.column_stack(tg), np.column_stack(tgA), np.column_stack(tgB)
    Ystk = C.zcols(np.column_stack([Yk[:, cols], Tm])); _, SB = C.split_half_reliability(np.column_stack([YAk[:, cols], TmA]), np.column_stack([YBk[:, cols], TmB]))
    CB = G["CB"][np.where(ok)[0]]; Bk, LG, _ = L.tx_blocks(S, CB, G["ncell"][ok], None)
    X0 = np.column_stack([np.asarray(v, float) for v in Bk.values()]); X0 = X0[:, X0.std(0) > 1e-9]; K = kernel(xy, a.sigma_ctx)
    TC = C.zcols(np.column_stack([X0, K @ X0])); TC = TC[:, TC.std(0) > 1e-9]; folds = C.spatial_folds(xy, 1000, seed=R)
    Yh = C.cv_ridge_gcv(TC, Ystk, folds); r2 = C.r2_columns(Ystk, Yh); E = Ystk - Yh
    ratio = np.where(SB >= a.min_sb, np.clip(r2, -1, 1) / SB, np.nan); nst = len(cols)
    base = ratio[:nst][np.isfinite(ratio[:nst])]; base_r2 = r2[:nst]
    # genes
    gn = np.asarray(S["genes"]); gpos = {g: i for i, g in enumerate(gn)}
    Zg = C.zcols(LG); mexp = LG.mean(0); detg = (LG > 0).mean(0)
    qb = np.quantile(mexp[mexp > 0], np.linspace(0, 1, 11)[1:-1]); gbin = np.digitize(mexp, qb)
    # lesion mirror pairs
    I = J = None; lab = (S.get("tx_labels") or {}).get("lesion")
    if lab is not None:
        kt = np.floor((S["tx_xy"] - G["o"]) / R).astype(np.int64); it = G["gi"].reindex(kt[:, 1] * 100000 + kt[:, 0]).fillna(-1).astype(int).values
        ls = pd.Series(np.asarray(lab).astype(str)).str.lower()
        ni = np.bincount(it[(it >= 0) & ls.str.startswith("intact").values], minlength=len(G["keys"]))[ok]
        nl = np.bincount(it[(it >= 0) & ls.str.startswith("lesion").values], minlength=len(G["keys"]))[ok]
        blab = np.where(ni > nl, "intact", np.where(nl > ni, "lesioned", "NA"))
        CTs = S["counts"]; tot = np.asarray(CTs.sum(1)).ravel(); LGs = (sp.diags(1e4 / np.maximum(tot, 1)) @ CTs).tocsr(); LGs.data = np.log1p(LGs.data)
        v_ = np.asarray(LGs.power(2).mean(0)).ravel() - np.asarray(LGs.mean(0)).ravel() ** 2; hv = np.argsort(-v_)[:2000]
        Zs = C.zcols(LGs[:, hv].toarray()); U, Sv, _ = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
        th, d, msc = C.mirror_axis(S["tx_xy"], U[:, :10] * Sv[:10])
        dd, jj = cKDTree(xy).query(C.reflect(xy, th, d, S["tx_xy"].mean(0)))
        Ii = np.where((blab == "intact") & (dd < 0.75 * R) & (blab[jj] == "lesioned"))[0]
        if len(Ii) >= 30:
            I, J = Ii, jj[Ii]
    for k_, info in enumerate(tginfo):
        j = nst + k_; mol = info["molecule"]; genes = C.NEURO_PANEL[mol][2] or []
        gi = [gpos[g] for g in genes if g in gpos]
        rho = p_g = local = np.nan
        if gi and Zg[:, gi].mean(1).std() > 1e-9:          # constant score if the synthesis genes are hardly expressed in this section -> correlation undefined (NaN, not tested)
            sc = Zg[:, gi].mean(1); rho = float(np.corrcoef(sc, Ystk[:, j])[0, 1]); local = float(np.mean(detg[gi]))
            null = [np.corrcoef(Zg[:, [rng.choice(np.where(gbin == gbin[g])[0]) for g in gi]].mean(1), Ystk[:, j])[0, 1] for _ in range(a.n_rand)]
            null = np.array(null)[np.isfinite(null)]
            p_g = float((np.sum(null >= rho) + 1) / (len(null) + 1)) if np.isfinite(rho) and len(null) else np.nan
        elif gi:
            local = float(np.mean(detg[gi]))
        # attribution: synthesis-gene score vs target-cell (medium spiny neuron) score, separate spatial-block cross-validated R²
        mi = [gpos[g] for g in C.MSN_TARGET_GENES if g in gpos]
        r2_syn = r2_msn = r2_both = np.nan; dmsn = np.nan
        if gi and mi:
            ssc, msc_ = C.zcols(Zg[:, gi].mean(1)[:, None]), C.zcols(Zg[:, mi].mean(1)[:, None]); yj = Ystk[:, [j]]
            r2_syn = float(C.r2_columns(yj, C.cv_ridge_gcv(ssc, yj, folds))[0]); r2_msn = float(C.r2_columns(yj, C.cv_ridge_gcv(msc_, yj, folds))[0])
            r2_both = float(C.r2_columns(yj, C.cv_ridge_gcv(np.column_stack([ssc, msc_]), yj, folds))[0])
        rr = dict(sample=S["sample"], section=sec, brain_area=area, mouse=S.get("mouse", ""), n_bins=int(ok.sum()), **info, SB=SB[j], R2_TC=r2[j], ratio=ratio[j],
                  pct_rank_ratio=float(np.mean(base < ratio[j])) if np.isfinite(ratio[j]) and len(base) else np.nan, pct_rank_R2=float(np.mean(base_r2 < r2[j])),
                  median_ratio_section=float(np.median(base)) if len(base) else np.nan, synthesis_genes=",".join(g for g in genes if g in gpos),
                  rho_synthesis=rho, p_synthesis=p_g, local_mrna_detection=local, R2_synthesis_only=r2_syn, R2_target_cells_only=r2_msn, R2_synthesis_plus_target=r2_both)
        if I is not None:
            DY = float((Ystk[I, j] - Ystk[J, j]).mean()); DE = float((E[I, j] - E[J, j]).mean())
            rr.update(n_pairs=len(I), delta_Y=DY, delta_pred=DY - DE, frac_asym_orthogonal=DE / DY if abs(DY) > 0.05 else np.nan,
                      delta_synthesis_genes=float((Zg[I][:, gi].mean(1) - Zg[J][:, gi].mean(1)).mean()) if gi else np.nan,
                      delta_target_cell_genes=float((Zg[I][:, mi].mean(1) - Zg[J][:, mi].mean(1)).mean()) if gi and mi else np.nan)
        rows.append(rr)
        if mol in ("dopamine", "GABA"):
            maps[(mol, S["sample"])] = (xy, Ystk[:, j], Zg[:, gi].mean(1) if gi else None, area)
    C.log(f"  {S['sample']} ({area}, {S.get('mouse', '')}): {ok.sum()} bins; {len(tginfo)} molecules; " + "; ".join(
        f"{r['molecule']} percentile {r['pct_rank_ratio']:.2f} r_synthesis {r['rho_synthesis']:+.2f} (p {r['p_synthesis']:.3f})" for r in rows[-len(tginfo):]
        if r["molecule"] in ("dopamine", "norepinephrine", "GABA", "serotonin")))
if not rows:
    C.log("no FMP-10 sections; D54 undetermined"); sys.exit(0)
T = pd.DataFrame(rows); T.to_csv(os.path.join(TAB, "m18_molecule_section.tsv"), sep="\t", index=False)
C.log("molecule x section:\n" + T.drop(columns=["ions"]).round(3).to_string(index=False))
v = {}
def area_stats(mol, ar):
    d = T[(T.molecule == mol) & (T.brain_area == ar)]
    return dict(n=int(len(d)), pct=float(d.pct_rank_ratio.median()) if len(d) else np.nan, frac_syn_sig=float((d.p_synthesis < 0.05).mean()) if len(d) else np.nan,
                rho=float(d.rho_synthesis.median()) if len(d) else np.nan, local_mrna=float(d.local_mrna_detection.median()) if len(d) else np.nan,
                frac_orth=float(d.frac_asym_orthogonal.median()) if "frac_asym_orthogonal" in d and d.frac_asym_orthogonal.notna().any() else np.nan)
for mol in ("dopamine", "norepinephrine", "serotonin", "GABA", "glutamate"):
    v[mol] = {ar: area_stats(mol, ar) for ar in ("Substantia Nigra", "Striatum")}
sn, st = v["dopamine"]["Substantia Nigra"], v["dopamine"]["Striatum"]
v["D54_soma_vs_terminal"] = bool(sn["n"] and st["n"] and (sn["pct"] - st["pct"] >= 0.2) and sn["frac_syn_sig"] >= 2 / 3 and st["frac_syn_sig"] <= 1 / 3) if sn["n"] and st["n"] else None
v["D54a_pct_diff"] = sn["pct"] - st["pct"] if sn["n"] and st["n"] else np.nan
da = T[T.molecule == "dopamine"]
if len(da) and da.R2_synthesis_only.notna().any():
    s_ = da[da.brain_area == "Substantia Nigra"]; t_ = da[da.brain_area == "Striatum"]
    v["D55_attribution"] = bool((s_.R2_synthesis_only > s_.R2_target_cells_only).mean() >= 2 / 3 and (t_.R2_target_cells_only > t_.R2_synthesis_only).mean() >= 2 / 3)
    v["D55_detail"] = {ar: dict(R2_syn=float(d_.R2_synthesis_only.median()), R2_target=float(d_.R2_target_cells_only.median())) for ar, d_ in (("Substantia Nigra", s_), ("Striatum", t_))}
ht = T[T.molecule == "serotonin"]; v["neg_control_5HT_no_gene_association"] = bool((ht.p_synthesis >= 0.05).all()) if len(ht) and ht.p_synthesis.notna().any() else None
g_ = T[T.molecule == "GABA"]; v["D54c_GABA_control"] = bool((g_.p_synthesis < 0.05).mean() >= 2 / 3) if len(g_) else None
v["D54d_lesion_orthogonal_str_gt_sn"] = bool(st["frac_orth"] > sn["frac_orth"]) if np.isfinite(st["frac_orth"]) and np.isfinite(sn["frac_orth"]) else None
C.save_json(v, os.path.join(TAB, "m18_verdict.json")); C.log("decision: " + json.dumps(v, ensure_ascii=False, default=str))
# figure: molecule x region
mols = [m for m in ("dopamine", "norepinephrine", "serotonin", "GABA", "glutamate", "taurine", "glutamine", "tyrosine") if (T.molecule == m).any()]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for k_, (col, lab) in enumerate((("pct_rank_ratio", "transcriptome predictability\n(percentile among features)"), ("rho_synthesis", "r with synthesis-gene score"),
                                 ("frac_asym_orthogonal", "lesion asymmetry: transcriptome-\northogonal fraction"))):
    for i, m in enumerate(mols):
        for ar, c_, dx in (("Substantia Nigra", "#de2d26", -.15), ("Striatum", "#3182bd", .15)):
            d = T[(T.molecule == m) & (T.brain_area == ar)]
            if col in d and d[col].notna().any():
                axs[k_].scatter(np.full(len(d), i + dx), d[col], c=c_, s=16, label=ar if i == 0 else None)
    axs[k_].set_xticks(range(len(mols))); axs[k_].set_xticklabels(mols, rotation=45, ha="right"); axs[k_].set_ylabel(lab)
axs[0].legend(fontsize=6, frameon=False); axs[0].set_title("same molecule, different compartment (red = SN, blue = striatum)", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m18_soma_terminal.png"), dpi=180); plt.close(fig)
for mol in ("dopamine", "GABA"):
    ks = [k for k in maps if k[0] == mol]
    if not ks:
        continue
    fig, axs = plt.subplots(2, len(ks), figsize=(2.4 * len(ks), 4.6), squeeze=False)
    for c_, k in enumerate(sorted(ks, key=lambda z: maps[z][3])):
        xy, yv, gv, ar = maps[k]
        axs[0, c_].scatter(xy[:, 0], xy[:, 1], c=yv, s=4, marker="s", cmap="magma"); axs[0, c_].set_title(f"{k[1][:14]}\n{ar}: {mol}", fontsize=6)
        if gv is not None:
            axs[1, c_].scatter(xy[:, 0], xy[:, 1], c=gv, s=4, marker="s", cmap="viridis"); axs[1, c_].set_title("synthesis-gene score", fontsize=6)
        for ax_ in axs[:, c_]:
            ax_.invert_yaxis(); ax_.set_aspect("equal"); ax_.set_xticks([]); ax_.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m18_maps_{mol}.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
