#!/usr/bin/env python3

import argparse, glob, os, re, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="IDHmut,ccRCC,GBM"); ap.add_argument("--res", default="200,400")
ap.add_argument("--n-null", type=int, default=19); ap.add_argument("--max-feat", type=int, default=60); ap.add_argument("--min-det", type=float, default=0.10)
ap.add_argument("--cluster-r", type=float, default=0.9); ap.add_argument("--block-um", type=float, default=1000.0)
ap.add_argument("--ccrcc-dir", default=None); ap.add_argument("--gmt", default=None); ap.add_argument("--ion-set", default="strict")
ap.add_argument("--include-uncertain", action="store_true"); ap.add_argument("--min-bins", type=int, default=80)
ap.add_argument("--step", default="Step6", help="output to 05_tables/<step>; run again in Step 7 with --step Step7 --n-null 99 (pooled null, D29 testable)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step)
plt.rcParams.update({"font.size": 7}); t0 = time.time(); rng = np.random.default_rng(0)
RES = [int(r) for r in a.res.split(",")]; DS = [d.strip() for d in a.datasets.split(",") if d.strip()]
SIGDIR = os.path.join(P["PROC"], "signatures")
gmt = a.gmt or next(iter(sorted(glob.glob(os.path.join(P["REFDB"], "h.all.*.symbols.gmt")))), None)
HALL = C.read_gmt(gmt) if gmt and os.path.exists(gmt) else None
STATE_FALLBACK = {k: v for k, v in {**C.PROGRAMS, **C.METAB_PROGRAMS}.items()
                  if k not in ("WM_oligodendrocyte", "Neuron_cortex", "Microglia", "Macrophage_TAM", "Vascular", "T_cell")}
C.log(f"state gene sets: {'MSigDB Hallmark (' + str(len(HALL)) + ' sets, ' + os.path.basename(gmt) + ')' if HALL else 'cell-state + metabolic programmes of p2_common (no Hallmark .gmt found; run p2_step6_downloads.sh)'}")
LM = None
lmf = os.path.join(P["REFDB"], "_parsed_LIPIDMAPS.tsv.gz")
if os.path.exists(lmf):
    LM = pd.read_csv(lmf, sep="\t")


def annotate(mz, pol, tol=5.0):
    """Candidate annotations from the built-in library and LIPID MAPS (reporting only, not used in the analysis)."""
    if not np.isfinite(mz) or pol not in ("positive", "negative"):
        return ""
    t = C.build_ion_table(pol); hit = t[np.abs(t.mz - mz) / mz * 1e6 < tol]
    names = [f"{r.name} {r.adduct}" for r in hit.itertuples()][:2]
    if LM is not None:
        ads = {"positive": {"[M+H]+": C.PROTON, "[M+Na]+": C.ELEM["Na"] - C.E_MASS, "[M+K]+": C.ELEM["K"] - C.E_MASS},
               "negative": {"[M-H]-": -C.PROTON, "[M+Cl]-": C.ELEM["Cl"] + C.E_MASS}}[pol]
        for ad_, dm in ads.items():
            m = LM[np.abs(LM.mono_mass + dm - mz) / mz * 1e6 < tol]
            nm_ = [x for x in m.name.astype(str) if not re.search(r"(-d\d|\(d\d|\bd\d+\b|13C|15N)", x)]   # drop isotope-labelled standards
            names += [f"{x} {ad_}" for x in nm_[:2]]
    return "; ".join(list(dict.fromkeys(names))[:3])


# ------------------------------------------------------------------ input: dict(dataset, sample, levels -> bins); for Visium the four predictor blocks are built at bin level
def visium_blocks(counts, genes, inv, sig, programs_fallback):
    nb = inv.max() + 1
    Bm = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(nb, len(inv)))
    CB = (Bm @ counts).tocsr() if sp.issparse(counts) else sp.csr_matrix(Bm @ counts)
    tot = np.asarray(CB.sum(1)).ravel(); g = pd.Index(genes)
    LG = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CB).toarray())
    ribo = np.asarray(CB[:, np.where(g.str.match(r"^RP[LS]\d"))[0]].sum(1)).ravel() / np.maximum(tot, 1)
    mito = np.asarray(CB[:, np.where(g.str.match(r"^(MT-|MTRNR2L)"))[0]].sum(1)).ravel() / np.maximum(tot, 1)
    tech = np.column_stack([np.log1p(tot), np.log1p(np.asarray((CB > 0).sum(1)).ravel()), ribo, mito])
    if sig is not None:
        comp, ncom = C.nnls_deconvolve(CB, genes, sig); comp = comp.loc[:, comp.std() > 1e-4]; src = f"NNLS deconvolution ({comp.shape[1]} types, {ncom} genes)"
    else:
        comp, _ = C.score_programs(LG, g, programs_fallback, min_genes=3); src = "marker programmes (no reference signature)"
    st_, _ = C.score_programs(LG, g, HALL if HALL else STATE_FALLBACK, min_genes=10 if HALL else 3)
    hv = np.argsort(-LG.var(0))[:2000]; Gp = PCA(30, random_state=0).fit_transform(StandardScaler().fit_transform(LG[:, hv]))
    return dict(tech=tech, comp=comp.values, state=st_.values, resid=Gp), comp.columns.tolist(), st_.columns.tolist(), src, Bm


def load_samples():
    out = []
    if "IDHmut" in DS:
        GT = "" if a.ion_set == "all" else a.ion_set + "_"
        for R in RES:
            f = os.path.join(P["PROC"], f"m4_grid_IDHmut_{GT}{R}um.parquet")
            if not os.path.exists(f):
                C.log(f"[IDH-mut] {f} missing (run run_step4.sh 0 first)"); continue
            G = pd.read_parquet(f)
            mc = [c for c in G if c.startswith("m:")]; cc = [c for c in G if c.startswith("ct:")]; pc = [c for c in G if c.startswith("prog:")]
            gc = [c for c in G if c.startswith("g:") and G[c].std() > 0]
            stc = [c for c in pc if c[5:] in STATE_FALLBACK]
            Gp = PCA(30, random_state=0).fit_transform(StandardScaler().fit_transform(G[gc].values))
            out.append(dict(dataset="IDH-mut", sample="SBIAD1426", res=R, xy=np.column_stack([G.gx.values, G.gy.values]) * R,
                            Y=G[mc].values, names=[c[2:] for c in mc], mz=np.full(len(mc), np.nan), pol=np.array(["positive"] * len(mc)),
                            blocks=dict(tech=np.log1p(G.ncell.values)[:, None], comp=G[cc].values, state=G[stc].values, resid=Gp),
                            comp_names=[c[3:] for c in cc], state_names=[c[5:] for c in stc], comp_src="Xenium cell-type fractions", annotated=True))
    if "ccRCC" in DS:
        d = a.ccrcc_dir or os.path.join(P["PUB"], "SpatialMETA2025_ccRCC_ST_SM")
        sig = pd.read_csv(os.path.join(SIGDIR, "ccRCC_signature.tsv"), sep="\t", index_col=0) if os.path.exists(os.path.join(SIGDIR, "ccRCC_signature.tsv")) else None
        for f in sorted(glob.glob(os.path.join(d, "adata_joint_*_raw.h5ad"))):
            s = os.path.basename(f).replace("adata_joint_", "").replace("_raw.h5ad", "")
            J = ad.read_h5ad(f); typ = pd.Series(J.var["type"].astype(str).values)
            sm, st = typ.str.startswith("SM").values, (typ == "ST").values
            X = J.X.tocsc(); SM = X[:, np.where(sm)[0]].toarray().astype(float); CT = X[:, np.where(st)[0]].tocsr()
            names = J.var_names[sm].astype(str).values; pol = np.array(["unknown"] * len(names), dtype=object)
            fsm = os.path.join(d, f"adata_SM_{s}_raw.h5ad")
            if os.path.exists(fsm):
                S_ = ad.read_h5ad(fsm, backed="r"); mp = dict(zip(S_.var_names.astype(str), S_.var["type"].astype(str))); S_.file.close()
                pol = np.array([{"SM_pos": "positive", "SM_neg": "negative"}.get(mp.get(n, ""), "unknown") for n in names], dtype=object)
            tic = SM.sum(1); keep = (tic > 0) & (np.asarray(CT.sum(1)).ravel() > 0)
            xy = np.asarray(J.obsm["spatial"], float)[keep]; xy = xy * 100.0 / C.spot_pitch(xy)
            gn = J.var["name"].astype(str).values[st] if "name" in J.var else J.var_names[st].astype(str).values
            for R in RES:
                inv, ctr = C.bin_spots(xy, R)
                if inv.max() + 1 < a.min_bins:
                    continue
                blocks, cn, sn, src, Bm = visium_blocks(CT[keep], gn, inv, sig, C.CCRCC_PROGRAMS)
                SMk = SM[keep] / tic[keep, None] * np.median(tic[keep]); cnt = np.asarray(Bm.sum(1)).ravel()
                SMb = (Bm @ SMk) / cnt[:, None]
                blocks["tech"] = np.column_stack([blocks["tech"], np.log1p((Bm @ tic[keep]) / cnt)])
                out.append(dict(dataset="ccRCC", sample=s, res=R, xy=ctr, Y=SMb, names=names, mz=pd.to_numeric(pd.Series(names), errors="coerce").values, pol=pol,
                                blocks=blocks, comp_names=cn, state_names=sn, comp_src=src, raw_intensity=True))
    if "GBM" in DS:
        IDT, allowed = C.ravi_identity(P, a.include_uncertain)
        sig = pd.read_csv(os.path.join(SIGDIR, "GBM_signature.tsv"), sep="\t", index_col=0) if os.path.exists(os.path.join(SIGDIR, "GBM_signature.tsv")) else None
        for pid in sorted(allowed):
            f = os.path.join(P["PROC"], "ravi_paired", f"{pid}_visium.h5ad"); fv = os.path.join(P["PROC"], "ravi_paired", f"{pid}_metab_var.tsv")
            if not (os.path.exists(f) and os.path.exists(fv)):
                continue
            B = ad.read_h5ad(f); var = pd.read_csv(fv, sep="\t", index_col=0)
            ok = B.obs.has_maldi.values.astype(bool) & (B.obs.n_maldi_px.values >= 1); m = C.ion_mask(var, a.ion_set)
            SM = np.asarray(B.obsm["metab"])[ok][:, m].astype(float)
            CT = B.layers["counts"] if "counts" in B.layers else sp.csr_matrix(np.expm1(B.X.toarray() if sp.issparse(B.X) else B.X))
            CT = (CT.tocsr() if sp.issparse(CT) else sp.csr_matrix(CT))[np.where(ok)[0]]
            xy = np.asarray(B.obsm["spatial_um"], float)[ok]
            for R in RES:
                inv, ctr = C.bin_spots(xy, R)
                if inv.max() + 1 < a.min_bins:
                    continue
                blocks, cn, sn, src, Bm = visium_blocks(CT, pd.Index(B.var_names.astype(str)), inv, sig, C.PROGRAMS)
                cnt = np.asarray(Bm.sum(1)).ravel(); SMb = (Bm @ SM) / cnt[:, None]
                blocks["tech"] = np.column_stack([blocks["tech"], np.log1p((Bm @ np.asarray(B.obs.n_maldi_px.values[ok], float)) / cnt)])
                out.append(dict(dataset="GBM", sample=pid, res=R, xy=ctr, Y=SMb, names=(var.best_name.astype(str) + " " + var.best_adduct.astype(str)).values[m],
                                mz=(var.mz_cal if "mz_cal" in var else var.mz).values[m], pol=np.array(["negative"] * m.sum()), blocks=blocks, comp_names=cn,
                                state_names=sn, comp_src=src, raw_intensity=True, annotated=True))
    return out


def poly2(xy):
    u = (xy[:, 0] - xy[:, 0].mean()) / xy[:, 0].std(); v = (xy[:, 1] - xy[:, 1].mean()) / xy[:, 1].std()
    return np.column_stack([np.ones_like(u), u, v, u * u, v * v, u * v])


def detrend(M, T):
    return M - T @ np.linalg.lstsq(T, M, rcond=None)[0]


rows, crow, summ_src = [], [], {}
for S in load_samples():
    Y0 = S["Y"]
    if S.get("raw_intensity"):
        det = (Y0 > 0).mean(0); okf = det >= a.min_det
        L = np.log1p(Y0[:, okf]); L = (L - L.mean(0)) / (L.std(0) + 1e-9)
    else:
        okf = np.ones(Y0.shape[1], bool); L = (Y0 - Y0.mean(0)) / (Y0.std(0) + 1e-9)
    names = np.asarray(S["names"])[okf]; mzs = np.asarray(S["mz"], float)[okf]; pols = np.asarray(S["pol"])[okf]
    I, pI = C.moran_knn(L, S["xy"], k=6, n_perm=99, seed=S["res"]); qI = C.bh(pI); st_ = np.where(qI < 0.05)[0]
    if len(st_) < 3:
        C.log(f"  {S['dataset']} {S['sample']} @ {S['res']} µm: fewer than 3 structured features; skipped"); continue
    # merge redundant ions: ions with |r| >= cluster_r form a cluster (isotopes / adducts / fragments), representative = highest Moran's I
    Cr = np.corrcoef(L[:, st_].T); Cr = np.nan_to_num(Cr)
    lab = fcluster(linkage(squareform(1 - np.abs(Cr), checks=False), "average"), t=1 - a.cluster_r, criterion="distance") if len(st_) > 2 else np.arange(len(st_)) + 1
    reps = []
    for k in np.unique(lab):
        mem = st_[lab == k]; rep = mem[np.argmax(I[mem])]; reps.append(rep)
        crow.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=S["res"], cluster=int(k), representative=names[rep], n_members=len(mem),
                         members="; ".join(map(str, names[mem][:12]))))
    reps = np.array(sorted(reps, key=lambda j: -I[j]))[: a.max_feat]
    Y = L[:, reps]; clus_of = {j: lab[list(st_).index(j)] for j in reps}
    bid = (np.floor(S["xy"][:, 1] / a.block_um) * 1000 + np.floor(S["xy"][:, 0] / a.block_um)).astype(int); ub, bix = np.unique(bid, return_inverse=True)
    folds = np.random.default_rng(S["res"]).permutation(np.arange(len(ub)) % 5)[bix]; blocks_idx = [np.where(bix == b)[0] for b in range(len(ub))]


    def bperm(M_):
        o = rng.permutation(len(ub)); out = np.empty_like(M_)
        for b in range(len(ub)):
            out[blocks_idx[b]] = M_[np.resize(blocks_idx[o[b]], len(blocks_idx[b]))]
        return out


    summ_src[f"{S['dataset']}|{S['sample']}"] = S["comp_src"]
    C.log(f"  {S['dataset']} {S['sample']} @ {S['res']} µm: {len(S['xy'])} bins; {len(st_)} structured features -> {len(np.unique(lab))} redundancy clusters -> {len(reps)} analysed; "
          f"composition = {S['comp_src']}; {S['blocks']['state'].shape[1]} state sets; {len(ub)} spatial blocks")
    for level in ("full", "detrended"):
        T = poly2(S["xy"])
        Yl = Y if level == "full" else detrend(Y, T)
        Bl = {k: (v if level == "full" else detrend(np.asarray(v, float), T)) for k, v in S["blocks"].items()}
        keepm = {k: np.asarray(v).std(0) > 1e-9 for k, v in Bl.items()}
        Bl = {k: np.asarray(v)[:, keepm[k]] for k, v in Bl.items()}
        phi, full, R2s = C.shapley_r2(Bl, Yl, folds)
        uniq, pv = {}, {}
        for b in ("state", "resid", "comp"):
            others = tuple(x for x in Bl if x != b); base = R2s[others]
            uniq[b] = full - base; nul = []
            for _ in range(a.n_null):
                Bp = dict(Bl); Bp[b] = bperm(Bl[b])
                nul.append(C.r2_columns(Yl, C.cv_multi_ridge(np.column_stack([Bp[x] for x in Bl]), Yl, folds)) - base)
            nul = np.array(nul); pv[b] = (np.sum(nul >= uniq[b][None, :], 0) + 1) / (a.n_null + 1); uniq[b + "_null_med"] = np.median(nul, 0)
            # pooled null: each feature's null values are standardized by their own median / MAD and pooled across features -> P resolution 1/(n_null x features)
            # (with n_null = 19 per feature, P >= 0.05 and BH q could never fall below 0.05)
            med = np.median(nul, 0); mad = 1.4826 * np.median(np.abs(nul - med), 0) + 1e-4
            zo = (uniq[b] - med) / mad; zn = np.sort(((nul - med) / mad).ravel())
            pv[b + "_pooled"] = (len(zn) - np.searchsorted(zn, zo, side="left") + 1) / (len(zn) + 1); uniq[b + "_z"] = zo
        # internal MALDI reference: first 15 PCs of the other cluster representatives (own cluster excluded)
        r_o = np.empty(Yl.shape[1])
        for j in range(Yl.shape[1]):
            oth = [i for i in range(Yl.shape[1]) if i != j]
            Op = PCA(min(15, len(oth)), random_state=0).fit_transform(Yl[:, oth]) if len(oth) > 15 else Yl[:, oth]
            r_o[j] = C.r2_columns(Yl[:, [j]], C.cv_multi_ridge(Op, Yl[:, [j]], folds))[0]
        # interpretation of unique state contributions: partial correlation with each state gene set given tech + comp
        Z = np.column_stack([np.ones(len(Yl)), Bl["tech"], Bl["comp"]])
        Yr = Yl - Z @ np.linalg.lstsq(Z, Yl, rcond=None)[0]; Sr = Bl["state"] - Z @ np.linalg.lstsq(Z, Bl["state"], rcond=None)[0]
        snames = [n for n, keep_ in zip(S["state_names"], keepm["state"]) if keep_]
        Rc = ((Yr - Yr.mean(0)) / (Yr.std(0) + 1e-9)).T @ ((Sr - Sr.mean(0)) / (Sr.std(0) + 1e-9)) / len(Yr)
        for j, col in enumerate(reps):
            top = np.argsort(-np.abs(Rc[j]))[:3]
            rows.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=S["res"], level=level, feature=names[col], mz=mzs[col], polarity=pols[col],
                             annotation=names[col] if S.get("annotated") else annotate(mzs[col], pols[col]), cluster_size=int((lab == clus_of[col]).sum()),
                             moran_I=float(I[col]), R2_full=full[j], **{f"shapley_{b}": float(phi[b][j]) if b in phi else np.nan for b in ("tech", "comp", "state", "resid")},
                             unique_state=uniq["state"][j], unique_state_adj=uniq["state"][j] - uniq["state_null_med"][j], p_state=pv["state"][j],
                             unique_resid=uniq["resid"][j], unique_resid_adj=uniq["resid"][j] - uniq["resid_null_med"][j], p_resid=pv["resid"][j],
                             unique_comp=uniq["comp"][j], p_comp=pv["comp"][j], R2_other_metabolites=r_o[j],
                             p_state_pooled=pv["state_pooled"][j], z_state=uniq["state_z"][j], p_resid_pooled=pv["resid_pooled"][j], p_comp_pooled=pv["comp_pooled"][j], n_null=a.n_null,
                             top_state_sets="; ".join(f"{snames[t]} ({Rc[j, t]:+.2f})" for t in top if t < len(snames))))
        d_ = pd.DataFrame([r for r in rows if r["dataset"] == S["dataset"] and r["sample"] == S["sample"] and r["res_um"] == S["res"] and r["level"] == level])
        C.log(f"     [{level}] median full-model R² {d_.R2_full.median():+.3f} = tech {d_.shapley_tech.median():+.3f} + composition {d_.shapley_comp.median():+.3f} + "
              f"state {d_.shapley_state.median():+.3f} + remaining {d_.shapley_resid.median():+.3f}; unique state (null-corrected) {d_.unique_state_adj.median():+.3f}; internal MALDI {d_.R2_other_metabolites.median():+.3f}")
D = pd.DataFrame(rows)
pd.DataFrame(crow).to_csv(os.path.join(TAB, "m7_feature_clusters.tsv"), sep="\t", index=False)
if not len(D):
    raise SystemExit("no results")
for b in ("state", "resid", "comp"):
    D[f"q_{b}"] = D.groupby(["dataset", "sample", "res_um", "level"])[f"p_{b}"].transform(lambda v: C.bh(v.values))
    D[f"q_{b}_pooled"] = D.groupby(["dataset", "sample", "res_um", "level"])[f"p_{b}_pooled"].transform(lambda v: C.bh(v.values))
D["meaningful_state_perfeature"] = (D.q_state < 0.05) & (D.unique_state_adj >= 0.02)          # per-feature null (not attainable with small n_null)
D["meaningful_state"] = (D.q_state_pooled < 0.05) & (D.unique_state_adj >= 0.02)               # pooled null
D["meaningful_resid"] = (D.q_resid_pooled < 0.05) & (D.unique_resid_adj >= 0.02)
pos = D.R2_full.clip(lower=1e-6)
for b in ("tech", "comp", "state", "resid"):
    D[f"frac_{b}"] = np.where(D.R2_full > 0.05, D[f"shapley_{b}"] / pos, np.nan)
D.to_csv(os.path.join(TAB, "m7_shapley.tsv"), sep="\t", index=False)
SM_ = D.groupby(["dataset", "sample", "res_um", "level"]).agg(
    n=("feature", "size"), R2_full=("R2_full", "median"), tech=("shapley_tech", "median"), comp=("shapley_comp", "median"),
    state=("shapley_state", "median"), resid=("shapley_resid", "median"), frac_tech=("frac_tech", "median"), frac_comp=("frac_comp", "median"),
    frac_state=("frac_state", "median"), frac_resid=("frac_resid", "median"), unique_state_adj=("unique_state_adj", "median"),
    frac_meaningful_state=("meaningful_state", "mean"), frac_meaningful_resid=("meaningful_resid", "mean"),
    frac_meaningful_state_perfeature=("meaningful_state_perfeature", "mean"), R2_other=("R2_other_metabolites", "median"),
    R2_full_p90=("R2_full", lambda v: float(np.nanpercentile(v, 90))), n_R2_gt_0_3=("R2_full", lambda v: int((v > 0.3).sum()))).reset_index()
SM_["composition_source"] = (SM_["dataset"] + "|" + SM_["sample"]).map(summ_src)
SM_["D27_comp_ge_state"] = SM_.comp >= SM_.state; SM_["D28_tech_ge_20pct"] = SM_.frac_tech >= 0.2; SM_["D29_state_encoded"] = SM_.frac_meaningful_state >= 0.2
SM_.to_csv(os.path.join(TAB, "m7_shapley_summary.tsv"), sep="\t", index=False)
C.log("summary (median):\n" + SM_.round(3).to_string(index=False))

lv_list = ["full", "detrended"]
fig, axs = plt.subplots(1, 2, figsize=(11, 0.32 * max(6, SM_[SM_.level == "full"].shape[0]) + 1.6))
cols = {"tech": "#969696", "comp": "#3182bd", "state": "#e6550d", "resid": "#fdae6b"}
for axx, lv in zip(axs, lv_list):
    d = SM_[SM_.level == lv].sort_values(["dataset", "sample", "res_um"]).reset_index(drop=True); yy = np.arange(len(d))[::-1]
    left_p = np.zeros(len(d)); left_n = np.zeros(len(d))
    for b, c in cols.items():
        v = d[b].values; pos_ = np.clip(v, 0, None); neg_ = np.clip(v, None, 0)
        axx.barh(yy, pos_, left=left_p, color=c, label=b); axx.barh(yy, neg_, left=left_n, color=c, alpha=.5)
        left_p += pos_; left_n += neg_
    axx.scatter(d.R2_other, yy, marker="|", s=80, c="k", label="MALDI internal", zorder=3)
    axx.set_yticks(yy); axx.set_yticklabels([f"{r.dataset} · {r['sample']} · {r.res_um} µm" for _, r in d.iterrows()], fontsize=6)
    axx.axvline(0, c="k", lw=.4); axx.set_xlabel("median Shapley R² across metabolite clusters"); axx.set_title(f"Transcriptome → metabolome predictability ({lv})")
    axx.legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m7_shapley_stacked.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
