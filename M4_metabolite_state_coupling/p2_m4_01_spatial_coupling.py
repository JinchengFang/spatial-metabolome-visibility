#!/usr/bin/env python3

import argparse, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--res", default="100,200,400"); ap.add_argument("--primary", type=int, default=200)
ap.add_argument("--n-perm", type=int, default=1999); ap.add_argument("--block-um", type=float, default=1000.0)
ap.add_argument("--tumor-frac", type=float, default=0.5, help="within-tumour analysis: minimum malignant-cell fraction of a bin")
ap.add_argument("--min-cells", type=int, default=3, help="minimum cells per bin at 100 µm (scaled with bin area)")
ap.add_argument("--ion-set", default="all", choices=["all", "strict", "protonated"], help="all = tier A/B; strict = without isobaric ambiguity and in-source water loss")
ap.add_argument("--step", default="Step3", help="output to 05_tables/<step>, 04_figures/<step>")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
GTAG = "" if a.ion_set == "all" else a.ion_set + "_"
RES = [int(r) for r in a.res.split(",")]; PRIM = a.primary if a.primary in RES else RES[len(RES) // 2]
plt.rcParams.update({"font.size": 7}); t0 = time.time()

# ------------------------------------------------------------------ input
pix = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"))
tis = ~pix.obs["empty"].values.astype(bool); pix = pix[tis].copy()
var = pix.var.copy()
ann = pd.read_csv(os.path.join(P["TAB"], "sbiad1426_peak_annotation.tsv"), sep="\t").set_index("peak")
for c_ in ("tier", "use_downstream", "best_name", "best_adduct", "isotope_of", "ambiguous"):
    if c_ in ann:
        var[c_] = ann.reindex(var.index)[c_].values
use = var.use_downstream.fillna(False).astype(bool).values
X = pix.X.tocsr() if sp.issparse(pix.X) else sp.csr_matrix(pix.X)
X = X.astype(np.float64)
Xu = X[:, np.where(use)[0]]
tic = np.asarray(Xu.sum(1)).ravel(); rms = np.sqrt(np.asarray(Xu.multiply(Xu).mean(1)).ravel())
ok = tic > 0
sel = C.ion_mask(var, a.ion_set)
ions = var.index[sel]; metab = var.loc[ions, "best_name"].astype(str).values
M_LIST = sorted(set(metab))
pd.DataFrame(dict(ion=ions, metabolite=metab, adduct=var.loc[ions, "best_adduct"].values, tier=var.loc[ions, "tier"].values)) \
    .to_csv(os.path.join(TAB, "m4_ion_map.tsv"), sep="\t", index=False)
C.log(f"IDH-mut: {ok.sum()} tissue pixels; ion set '{a.ion_set}': {len(ions)} ions -> {len(M_LIST)} metabolites")
px_x, px_y = pix.obs.x_um.values, pix.obs.y_um.values
cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"))
cx, cy = cel.obsm["spatial"][:, 0], cel.obsm["spatial"][:, 1]
x0, y0 = min(px_x[ok].min(), cx.min()), min(px_y[ok].min(), cy.min())
anno = cel.obs["Anno"].astype(str).values; lin = cel.obs["lineage"].astype(str).values
vc = pd.Series(anno).value_counts(); ctype = np.where(pd.Series(anno).map(vc).values >= 100, anno, lin)
CT, LIN = sorted(np.unique(ctype)), sorted(np.unique(lin))
CT_LIN = pd.crosstab(ctype, lin).idxmax(1).to_dict()            # cell type -> lineage
MAL_STATES = [t for t in CT if CT_LIN.get(t) == "Malignant" and t != "Malignant"]
G = cel.X.tocsr() if sp.issparse(cel.X) else sp.csr_matrix(cel.X)
genes = pd.Index(cel.var_names)
ENZ_GENES = sorted({g for d in C.XENIUM_ENZ.values() for g in d if g in genes})

# main expressing cell type of each enzyme (single-cell level, for interpretation)
Gl = sp.diags(1e4 / np.maximum(np.asarray(G.sum(1)).ravel(), 1)) @ G; Gl.data = np.log1p(Gl.data)
src = []
for g in ENZ_GENES:
    v = np.asarray(Gl[:, genes.get_loc(g)].todense()).ravel()
    s = pd.Series(v).groupby(ctype).agg(["mean", lambda z: (z > 0).mean()]); s.columns = ["mean_logCPM", "frac_pos"]
    s = s.sort_values("mean_logCPM", ascending=False)
    src.append(dict(gene=g, top1=s.index[0], top1_mean=s.mean_logCPM.iloc[0], top2=s.index[1], top2_mean=s.mean_logCPM.iloc[1],
                    top3=s.index[2], frac_pos_top1=s.frac_pos.iloc[0], overall_frac_pos=float((v > 0).mean())))
pd.DataFrame(src).to_csv(os.path.join(TAB, "m4_enzyme_source_celltype.tsv"), sep="\t", index=False)


def bin_metab(R, norm):
    """Pixels -> R-µm bin means (norm = tic / rms) -> log1p per ion -> z -> mean over ions of the same metabolite."""
    d = norm[ok]; f = np.median(d) / d
    ix = np.floor((px_x[ok] - x0) / R).astype(int); iy = np.floor((px_y[ok] - y0) / R).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1; lin_ = iy * nx + ix
    Bm = sp.csr_matrix((f, (lin_, np.arange(len(lin_)))), shape=(nx * ny, len(lin_)))
    cnt = np.bincount(lin_, minlength=nx * ny)
    I = np.asarray((Bm @ X[ok][:, np.where(sel)[0]]).todense()) / np.maximum(cnt, 1)[:, None]
    return np.log1p(I), cnt, nx, ny


def bin_cells(R, nx, ny):
    ix = np.floor((cx - x0) / R).astype(int); iy = np.floor((cy - y0) / R).astype(int)
    keep = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny); lin_ = np.where(keep, iy * nx + ix, 0)
    n = np.bincount(lin_[keep], minlength=nx * ny).astype(float)
    feats = {}
    for t in CT:
        feats["ct:" + t] = np.bincount(lin_[keep & (ctype == t)], minlength=nx * ny) / np.maximum(n, 1)
    for t in LIN:
        feats["lin:" + t] = np.bincount(lin_[keep & (lin == t)], minlength=nx * ny) / np.maximum(n, 1)
    Bm = sp.csr_matrix((np.ones(keep.sum()), (lin_[keep], np.where(keep)[0])), shape=(nx * ny, len(cx)))
    GB = np.asarray((Bm @ G).todense()); LG = np.log1p(GB / np.maximum(GB.sum(1, keepdims=True), 1) * 1e4)
    return n, feats, LG


rows, store, TROWS, STRUCT = [], {}, [], []
for R in RES:
    Lt, cnt, nx, ny = bin_metab(R, tic)
    Lr, _, _, _ = bin_metab(R, rms)
    ncell, feats, LG = bin_cells(R, nx, ny)
    good = (cnt >= max(1, (R / 10.0) ** 2 / 3)) & (ncell >= a.min_cells * (R / 100.0) ** 2 * 0.5)
    gi = np.where(good)[0]; gx, gy = gi % nx, gi // nx
    S_prog, used = C.score_programs(LG[gi], genes, {**C.PROGRAMS, **C.METAB_PROGRAMS}, min_genes=2)
    F = pd.DataFrame({k: v[gi] for k, v in feats.items()})
    for c_ in S_prog:
        F["prog:" + c_] = S_prog[c_].values
    for g in ENZ_GENES:
        F["gene:" + g] = LG[gi, genes.get_loc(g)]
    F = F.loc[:, F.std() > 0]
    z = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-9)
    Zt, Zr = z(Lt[gi]), z(Lr[gi])
    Mt = pd.DataFrame({m: Zt[:, metab == m].mean(1) for m in M_LIST}); Mr = pd.DataFrame({m: Zr[:, metab == m].mean(1) for m in M_LIST})
    Mi = pd.DataFrame(Zt, columns=ions)
    cov = np.column_stack([F["lin:Malignant"].values if "lin:Malignant" in F else np.zeros(len(F)), np.log1p(ncell[gi])])
    # programmes / enzymes: adjusted for all main lineages (malignant, neuron, oligodendrocyte, myeloid, vascular, astrocyte + density) -> signal beyond cell composition
    cov_full = np.column_stack([F[c_].values for c_ in ("lin:Malignant", "lin:Neuron", "lin:Oligodendrocyte", "lin:Myeloid", "lin:Vascular", "lin:Astrocyte")
                                if c_ in F] + [np.log1p(ncell[gi])])
    blk = max(3, int(round(a.block_um / R)))
    C.log(f"[{R} µm] {len(gi)} bins x {Mt.shape[1]} metabolites x {F.shape[1]} cell features; blocks of {blk}x{blk} bins (about {blk * R} µm); {a.n_perm} permutations")
    r, p, sd = C.block_perm_corr(Mt.values, F.values, gx, gy, block=blk, n_perm=a.n_perm, seed=R)
    Fa = C.residualize_ranks(F.values, cov); Ma = C.residualize_ranks(Mt.values, cov)
    # the malignant fraction cannot be adjusted for itself -> adjusted for cell density only
    for j, c_ in enumerate(F.columns):
        if c_ == "lin:Malignant":
            Fa[:, j] = C.residualize_ranks(F[c_].values, cov[:, 1:])[:, 0]
    ra, pa, _ = C.block_perm_corr(Ma, Fa, gx, gy, block=blk, n_perm=a.n_perm, seed=R + 1, ranks=False)
    pf = [j for j, c_ in enumerate(F.columns) if c_.startswith(("prog:", "gene:"))]
    rf_, pf_, _ = C.block_perm_corr(C.residualize_ranks(Mt.values, cov_full), C.residualize_ranks(F.values[:, pf], cov_full), gx, gy,
                                    block=blk, n_perm=a.n_perm, seed=R + 2, ranks=False)
    rfull = np.full_like(r, np.nan); pfull = np.full_like(r, np.nan); rfull[:, pf] = rf_; pfull[:, pf] = pf_
    if R == PRIM:   # block-size sensitivity: blocks of twice the size (about 2 mm)
        _, p2b, _ = C.block_perm_corr(Mt.values, F.values, gx, gy, block=2 * blk, n_perm=a.n_perm, seed=R + 3)
    else:
        p2b = np.full_like(r, np.nan)
    rr_ = C._rank_cols(Mr.values); rf = C._rank_cols(F.values)
    rms_r = ((rr_ - rr_.mean(0)) / rr_.std(0)).T @ ((rf - rf.mean(0)) / (rf.std(0) + 1e-12)) / len(rf)
    ri = C._rank_cols(Mi.values)
    ion_r = ((ri - ri.mean(0)) / (ri.std(0) + 1e-12)).T @ ((rf - rf.mean(0)) / (rf.std(0) + 1e-12)) / len(rf)
    # ---- within tumour: in bins with malignant fraction >= 0.5, do metabolites track the state composition of malignant cells (AC / MES / OPC / NPC-like) and programmes?
    if "lin:Malignant" in F and MAL_STATES:
        tm = F["lin:Malignant"].values >= a.tumor_frac
        if tm.sum() >= 60:
            Ft = pd.DataFrame({"mstate:" + t: F["ct:" + t].values[tm] / np.maximum(F["lin:Malignant"].values[tm], 1e-9) for t in MAL_STATES if "ct:" + t in F})
            for c_ in F.columns:
                if c_.startswith("prog:"):
                    Ft[c_] = F[c_].values[tm]
            Ft = Ft.loc[:, Ft.std() > 0]
            ct_ = np.column_stack([F["lin:Malignant"].values[tm], np.log1p(ncell[gi][tm])])
            bt = max(3, int(round(a.block_um / R)))
            r_t, p_t, _ = C.block_perm_corr(C.residualize_ranks(Mt.values[tm], ct_), C.residualize_ranks(Ft.values, ct_), gx[tm], gy[tm],
                                            block=bt, n_perm=a.n_perm, seed=R + 4, ranks=False)
            rr_t = C.residualize_ranks(Mr.values[tm], ct_); rf_t = C.residualize_ranks(Ft.values, ct_)
            zz = lambda M_: (M_ - M_.mean(0)) / (M_.std(0) + 1e-12)
            rms_t = zz(rr_t).T @ zz(rf_t) / len(rf_t)
            for i, m in enumerate(M_LIST):
                for j, f_ in enumerate(Ft.columns):
                    TROWS.append(dict(res_um=R, metabolite=m, feature=f_, rho=r_t[i, j], p_perm=p_t[i, j], rho_rms=rms_t[i, j], n_bins=int(tm.sum())))
            C.log(f"   within tumour (malignant >= {a.tumor_frac:.0%}): {int(tm.sum())} bins x {Ft.shape[1]} malignant-state / programme features")
            # spatial structure of within-tumour metabolite residuals (malignant fraction + density removed): Moran's I with a permutation null
            Mres = C.residualize_ranks(Mt.values[tm], ct_); rng_ = np.random.default_rng(R)
            for i, m in enumerate(M_LIST):
                im_ = np.full((ny, nx), np.nan); im_[gy[tm], gx[tm]] = Mres[:, i]; I0 = C.morans_i_grid(im_)
                nul = []
                for _ in range(199):
                    im_[gy[tm], gx[tm]] = rng_.permutation(Mres[:, i]); nul.append(C.morans_i_grid(im_))
                # largest |rho| of the within-tumour residual with malignant states / programmes
                STRUCT.append(dict(res_um=R, metabolite=m, moran_I=I0, p_moran=(np.sum(np.array(nul) >= I0) + 1) / 200,
                                   max_abs_rho_state=float(np.nanmax(np.abs(r_t[i]))), best_state=Ft.columns[int(np.nanargmax(np.abs(r_t[i])))], n_bins=int(tm.sum())))
    for i, m in enumerate(M_LIST):
        idx = np.where(metab == m)[0]
        for j, f_ in enumerate(F.columns):
            sg = np.sign(r[i, j]); ir = ion_r[idx, j]
            rows.append(dict(res_um=R, metabolite=m, n_ions=len(idx), feature=f_, kind=f_.split(":")[0], rho=r[i, j], null_sd=sd[i, j], p_perm=p[i, j],
                             rho_adj=ra[i, j], p_adj=pa[i, j], rho_full=rfull[i, j], p_full=pfull[i, j], p_block2x=p2b[i, j], rho_rms=rms_r[i, j],
                             adduct_agree=float(np.mean(np.sign(ir) == sg)) if len(idx) > 1 else 1.0, n_bins=len(gi)))
    store[R] = dict(gx=gx, gy=gy, nx=nx, ny=ny, M=Mt, F=F)
    # save bin matrices for M5-01: coordinates / metabolites / ions / cell features / all panel genes
    GG = pd.DataFrame(LG[gi], columns=["g:" + g for g in genes])
    Out = pd.concat([pd.DataFrame(dict(gx=gx, gy=gy, ncell=ncell[gi])), Mt.add_prefix("m:"), Mr.add_prefix("mrms:"),
                     pd.DataFrame(Zt, columns=["ion:" + i_ for i_ in ions]), F, GG], axis=1)
    Out.attrs = {}
    Out.to_parquet(os.path.join(P["PROC"], f"m4_grid_IDHmut_{GTAG}{R}um.parquet"))
D = pd.DataFrame(rows)
D["family"] = np.where(D.kind == "gene", "enzyme", "state")
for (R, fam), g in D.groupby(["res_um", "family"]):
    D.loc[g.index, "q"] = C.bh(g.p_perm.values); D.loc[g.index, "q_adj"] = C.bh(g.p_adj.values)
    for c_, qc in (("p_full", "q_full"), ("p_block2x", "q_block2x")):
        v = g[c_].notna()
        D.loc[g.index[v], qc] = C.bh(g.loc[v, c_].values) if v.any() else np.nan
D["cptac_name"] = D.metabolite.map(C.CPTAC_NAME)
D.to_csv(os.path.join(TAB, "m4_coupling_IDHmut_all.tsv"), sep="\t", index=False)

# ------------------------------------------------------------------ summary across scales and pre-specified decisions
W = D.pivot_table(index=["metabolite", "feature", "kind", "family", "n_ions"], columns="res_um",
                  values=["rho", "q", "rho_adj", "q_adj", "rho_full", "q_full", "q_block2x", "rho_rms", "adduct_agree"], dropna=False).reset_index()
W.columns = [f"{a_}_{b_}" if b_ != "" else a_ for a_, b_ in W.columns]
W = W[W[f"rho_{PRIM if PRIM in RES else RES[0]}"].notna()].copy()     # pivot_table(dropna=False) creates the full Cartesian product; drop absent combinations
pr = lambda c_: f"{c_}_{PRIM}"
others = [R for R in RES if R != PRIM]
same = np.ones(len(W), bool)
for R in others:
    same &= (np.sign(W[f"rho_{R}"]) == np.sign(W[pr("rho")])) & (W[f"rho_{R}"].abs() >= 0.1)
W["robust"] = (W[pr("q")] < 0.05) & (W[pr("rho")].abs() >= 0.2) & same & (np.sign(W[pr("rho_rms")]) == np.sign(W[pr("rho")])) & (W[pr("adduct_agree")] >= 2 / 3 - 1e-9)
W["robust_block2x"] = W.robust & (W[pr("q_block2x")] < 0.05)
# composition-independent: cell types / lineages significant after adjustment for [malignant fraction, density]; programmes / enzymes after adjustment for all main lineages
fullk = W.kind.isin(["prog", "gene"])
W["composition_independent"] = W.robust & np.where(fullk, (W[pr("q_full")] < 0.05) & (np.sign(W[pr("rho_full")]) == np.sign(W[pr("rho")])),
                                                   (W[pr("q_adj")] < 0.05) & (np.sign(W[pr("rho_adj")]) == np.sign(W[pr("rho")])))
W["cptac_name"] = W.metabolite.map(C.CPTAC_NAME)
W = W.sort_values(["robust", pr("rho")], ascending=[False, False], key=lambda s: s.abs() if s.dtype != bool else s)
W.to_csv(os.path.join(TAB, "m4_coupling_IDHmut_summary.tsv"), sep="\t", index=False)
st = W[W.family == "state"]
C.log(f"state features: robust coupling {int(st.robust.sum())} / {len(st)}; significant with 2x blocks {int(st.robust_block2x.sum())}; composition-independent {int(st.composition_independent.sum())}")
C.log("  (by feature type) " + "; ".join(f"{k}: {int(g.robust.sum())}/{len(g)} robust, {int(g.composition_independent.sum())} composition-independent" for k, g in st.groupby("kind")))
for r_ in st[st.composition_independent & st.kind.isin(["prog"])].head(15).itertuples():
    C.log(f"   [programme, beyond composition] {r_.metabolite:22s} ~ {r_.feature:30s} rho{PRIM}={getattr(r_, 'rho_' + str(PRIM)):+.2f}  fully adjusted {getattr(r_, 'rho_full_' + str(PRIM)):+.2f}")
for r_ in st[st.robust & st.kind.isin(["ct"])].head(15).itertuples():
    C.log(f"   [cell type] {r_.metabolite:22s} ~ {r_.feature:30s} rho{PRIM}={getattr(r_, 'rho_' + str(PRIM)):+.2f}  adjusted {getattr(r_, 'rho_adj_' + str(PRIM)):+.2f}"
          f"{'  [composition-independent]' if r_.composition_independent else ''}")

# within-tumour malignant-state coupling
TT = pd.DataFrame(TROWS)
if len(TT):
    for R, g in TT.groupby("res_um"):
        TT.loc[g.index, "q"] = C.bh(g.p_perm.values)
    TW = TT.pivot_table(index=["metabolite", "feature"], columns="res_um", values=["rho", "q", "rho_rms", "n_bins"]).reset_index()
    TW.columns = [f"{a_}_{b_}" if b_ != "" else a_ for a_, b_ in TW.columns]
    if f"rho_{PRIM}" in TW:
        TW = TW[TW[f"rho_{PRIM}"].notna()].copy()
        same_t = np.ones(len(TW), bool)
        for R in others:
            if f"rho_{R}" in TW:   # not required at scales with too few bins (< 40)
                same_t &= TW[f"rho_{R}"].isna() | ((np.sign(TW[f"rho_{R}"]) == np.sign(TW[f"rho_{PRIM}"])) & (TW[f"rho_{R}"].abs() >= 0.1))
        TW["robust"] = (TW[f"q_{PRIM}"] < 0.05) & (TW[f"rho_{PRIM}"].abs() >= 0.2) & same_t & (np.sign(TW[f"rho_rms_{PRIM}"]) == np.sign(TW[f"rho_{PRIM}"]))
        TW["cptac_name"] = TW.metabolite.map(C.CPTAC_NAME)
        TW = TW.sort_values(["robust", f"q_{PRIM}"], ascending=[False, True])
        TW.to_csv(os.path.join(TAB, "m4_within_tumor_state_coupling.tsv"), sep="\t", index=False)
        C.log(f"within tumour, metabolite ~ malignant state / programme: robust {int(TW.robust.sum())} / {len(TW)} (all adjusted for malignant fraction + density)")
        for r_ in TW[TW.robust].head(15).itertuples():
            C.log(f"   [within tumour] {r_.metabolite:22s} ~ {r_.feature:30s} rho{PRIM}={getattr(r_, 'rho_' + str(PRIM)):+.2f}  q={getattr(r_, 'q_' + str(PRIM)):.2g}")
else:
    TW = pd.DataFrame()
SS = pd.DataFrame(STRUCT)
if len(SS):
    SS["q_moran"] = SS.groupby("res_um").p_moran.transform(lambda v: C.bh(v.values))
    # structured but decoupled from transcriptional state: significant within-tumour spatial structure (q < 0.05) and no robust malignant-state / programme coupling
    rob_m = set(TW[TW.robust].metabolite) if len(TW) and "robust" in TW else set()
    SS["structured_decoupled"] = (SS.q_moran < 0.05) & ~SS.metabolite.isin(rob_m)
    SS.to_csv(os.path.join(TAB, "m4_within_tumor_structure.tsv"), sep="\t", index=False)
    sp_ = SS[SS.res_um == PRIM]
    C.log(f"within tumour ({PRIM} µm): {int((sp_.q_moran < 0.05).sum())}/{len(sp_)} metabolites keep significant spatial structure after removing tumour content; "
          f"decoupled from malignant states / programmes: {int(sp_.structured_decoupled.sum())}: " + ", ".join(sp_[sp_.structured_decoupled].sort_values("moran_I", ascending=False).metabolite.head(10)))

# enzyme co-localization (pre-specified pairs)
E = []
for m, d in C.XENIUM_ENZ.items():
    for g, sgn in d.items():
        h = W[(W.metabolite == m) & (W.feature == "gene:" + g)]
        if len(h):
            h = h.iloc[0]
            E.append(dict(metabolite=m, gene=g, expected_sign=sgn, **{k: h[k] for k in W.columns if k.startswith(("rho_", "q_")) and not k.startswith("rho_rms")},
                          robust=bool(h.robust), composition_independent=bool(h.composition_independent)))
E = pd.DataFrame(E)
if len(E):
    E["q_family"] = C.bh(D[(D.res_um == PRIM)].set_index(["metabolite", "feature"]).reindex(list(zip(E.metabolite, "gene:" + E.gene))).p_perm.values)
    E["sign_as_expected"] = np.where(E.expected_sign == "+", E[pr("rho")] > 0, E[pr("rho")] < 0)
    E = E.sort_values("q_family"); E.to_csv(os.path.join(TAB, "m4_enzyme_colocalization.tsv"), sep="\t", index=False)
    for r_ in E.head(12).itertuples():
        C.log(f"   enzyme {r_.metabolite:20s} - {r_.gene:7s} expected {r_.expected_sign} rho={getattr(r_, 'rho_' + str(PRIM)):+.2f} q={r_.q_family:.2g} fully adjusted rho={getattr(r_, 'rho_full_' + str(PRIM)):+.2f}"
              f"{'  sign as expected' if r_.sign_as_expected else '  sign not as expected'}")

# export for the CPTAC replication: programme features + cell types (mapped to programmes)
R4 = W[(W.family == "state") & W.cptac_name.notna()].copy()
R4["program"] = np.where(R4.kind == "prog", R4.feature.str.replace("prog:", "", regex=False),
                         R4.feature.str.split(":").str[1].map(C.CELLTYPE_TO_PROGRAM))
R4 = R4[R4.program.notna() & R4.kind.isin(["prog", "ct"])]
R4 = R4.sort_values(pr("q")).drop_duplicates(["metabolite", "program"])
R4[["metabolite", "cptac_name", "feature", "program", pr("rho"), pr("q"), pr("rho_adj"), pr("q_adj"), "robust", "composition_independent"]] \
    .rename(columns={pr("rho"): "rho_spatial", pr("q"): "q_spatial", pr("rho_adj"): "rho_spatial_adj", pr("q_adj"): "q_spatial_adj"}) \
    .to_csv(os.path.join(TAB, "m4_couplings_for_cptac.tsv"), sep="\t", index=False)
C.log(f"(metabolite, programme) pairs testable in CPTAC: {len(R4)}, spatially robust {int(R4.robust.sum())}")

# ------------------------------------------------------------------ figures
S = store[PRIM]; Dp = D[(D.res_um == PRIM) & (D.family == "state") & (D.kind.isin(["ct", "prog"]))]
H = Dp.pivot(index="metabolite", columns="feature", values="rho"); Q = Dp.pivot(index="metabolite", columns="feature", values="q")
keep_c = H.columns[(Q < 0.05).any(axis=0)] if (Q < 0.05).any(axis=None) else H.columns
H, Q = H[keep_c], Q[keep_c]
if H.shape[0] > 1 and H.shape[1] > 1:
    from scipy.cluster.hierarchy import leaves_list, linkage
    oi = leaves_list(linkage(np.nan_to_num(H.values), "average")); oj = leaves_list(linkage(np.nan_to_num(H.values.T), "average"))
    H, Q = H.iloc[oi, oj], Q.iloc[oi, oj]
fig, ax = plt.subplots(figsize=(max(5, .28 * H.shape[1] + 2.5), max(3, .22 * H.shape[0] + 1.5)))
im = ax.imshow(H.values, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        q_ = Q.values[i, j]
        if q_ < 0.05:
            ax.text(j, i, "**" if q_ < 0.01 else "*", ha="center", va="center", fontsize=6)
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index); ax.set_xticks(range(H.shape[1]))
ax.set_xticklabels([c_.replace("prog:", "P:").replace("ct:", "") for c_ in H.columns], rotation=90)
plt.colorbar(im, ax=ax, shrink=.6, label=f"Spearman ρ ({PRIM} µm)"); ax.set_title(f"IDH-mut metabolite–cell-state coupling ({PRIM} µm; * q<0.05 block permutation)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"m4_coupling_heatmap_{PRIM}um.png"), dpi=160); plt.close(fig)

top = st[st.robust].head(12)
if len(top):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for r_ in top.itertuples():
        ax.plot(RES, [getattr(r_, f"rho_{R}") for R in RES], "-o", ms=3, label=f"{r_.metabolite}~{r_.feature.split(':')[1]}")
    ax.axhline(0, color="k", lw=.5); ax.set_xscale("log"); ax.set_xticks(RES); ax.set_xticklabels(RES); ax.set_xlabel("grid (µm)"); ax.set_ylabel("ρ")
    ax.legend(fontsize=5, frameon=False, bbox_to_anchor=(1, 1)); ax.set_title("Robust couplings across scales")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m4_coupling_scales.png"), dpi=160); plt.close(fig)
    show = top.head(4)
    fig, axs = plt.subplots(len(show), 2, figsize=(5.4, 2.5 * len(show)), squeeze=False)
    for k_, r_ in enumerate(show.itertuples()):
        for j_, (vals, ttl) in enumerate(((S["M"][r_.metabolite].values, r_.metabolite), (S["F"][r_.feature].values, r_.feature))):
            im_ = np.full((S["ny"], S["nx"]), np.nan); im_[S["gy"], S["gx"]] = vals
            axs[k_, j_].imshow(im_, cmap="magma", interpolation="nearest"); axs[k_, j_].axis("off"); axs[k_, j_].set_title(ttl, fontsize=7)
    fig.suptitle(f"Top robust couplings ({PRIM} µm)", fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, "m4_top_coupling_maps.png"), dpi=150); plt.close(fig)

C.save_json(dict(ion_set=a.ion_set, n_ions=int(len(ions)), res_um=RES, primary_um=PRIM, n_perm=a.n_perm, block_um=a.block_um, n_metabolites=len(M_LIST), metabolites=M_LIST,
                 n_state_features=int(st.feature.nunique()), n_robust=int(st.robust.sum()), n_comp_independent=int(st.composition_independent.sum()),
                 robust_pairs=[f"{r_.metabolite}~{r_.feature}" for r_ in st[st.robust].itertuples()],
                 within_tumor_robust=[f"{r_.metabolite}~{r_.feature}" for r_ in TW[TW.robust].itertuples()] if len(TW) and "robust" in TW else [],
                 within_tumor_structured_decoupled=sorted(SS[(SS.res_um == PRIM) & SS.structured_decoupled].metabolite) if len(SS) else [],
                 enzyme_pairs_full_adjusted_q05=int((E[pr("q_full")] < 0.05).sum()) if len(E) and pr("q_full") in E else 0,
                 enzyme_pairs_tested=len(E), enzyme_pairs_q05=int((E.q_family < 0.05).sum()) if len(E) else 0,
                 enzyme_pairs_q05_expected_sign=int(((E.q_family < 0.05) & E.sign_as_expected).sum()) if len(E) else 0,
                 runtime_s=round(time.time() - t0)), os.path.join(TAB, "m4_summary.json"))
C.log(f"done ({time.time() - t0:.0f} s) -> {TAB}")
