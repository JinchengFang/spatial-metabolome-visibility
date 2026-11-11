#!/usr/bin/env python3

import argparse, glob, json, os, re, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "M20_autonomy_spectrum"))
import p2_common as C
from m20_helpers import make_mapper, batch_X, batch_cv_r2
warnings.filterwarnings("ignore")
import anndata as ad

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--data-dir", default=None); ap.add_argument("--polarity", default="auto")
ap.add_argument("--k-grid", type=float, default=2.0); ap.add_argument("--grid-units", type=float, default=None, help="bin size in data coordinate units"); ap.add_argument("--target-bins", type=int, default=600); ap.add_argument("--n-rand", type=int, default=999); ap.add_argument("--n-null", type=int, default=499); ap.add_argument("--step", default="Step21")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(2101)
DD = a.data_dir or os.path.join(P["PUB"], "Li2026_COSIE_mouse_hippocampus_MALDI_Visium")
NAA = {"positive": {"[M+H]+": 176.0553, "[M+Na]+": 198.0373, "[M+K]+": 214.0113}, "negative": {"[M-H]-": 174.0408, "[M+Cl]-": 210.0175}}
NEUR_SIDE = ["NAT8L"] + C.PROGRAMS["Neuron_cortex"].split(); OLIG_SIDE = ["ASPA"] + C.PROGRAMS["WM_oligodendrocyte"].split()
inv = [f"external data set: {DD}"]


def mz_of(A):
    for c in ("mz", "m/z", "MZ", "mass"):
        if c in A.var:
            return pd.to_numeric(A.var[c], errors="coerce").values
    v = [re.search(r"(\d{2,4}\.\d{2,})", str(x)) for x in A.var_names]
    return np.array([float(m.group(1)) if m else np.nan for m in v])


def coords(A):
    for k in ("spatial", "X_spatial", "spatial_coords", "coords"):
        if k in A.obsm:
            return np.asarray(A.obsm[k], float)[:, :2]
    for cx, cy in (("x", "y"), ("X", "Y"), ("imagecol", "imagerow"), ("pxl_col_in_fullres", "pxl_row_in_fullres"), ("x_coord", "y_coord"), ("array_col", "array_row")):
        if cx in A.obs and cy in A.obs:
            return A.obs[[cx, cy]].values.astype(float)
    return None


def is_metab(A):
    m = mz_of(A); return np.isfinite(m).mean() > 0.8 and np.nanmedian(m) > 50


files = sorted(glob.glob(os.path.join(DD, "**", "*.h5ad"), recursive=True))
if not files:
    C.log(f"no .h5ad found in {DD} (run bash p2_step21_downloads.sh first); other files:"); [C.log("   " + f) for f in sorted(glob.glob(os.path.join(DD, "**", "*"), recursive=True))[:50]]
    sys.exit(0)
groups = {}
for f in files:
    A = ad.read_h5ad(f, backed="r"); xy = coords(A); kind = "metab" if is_metab(A) else "rna"
    nn = float(np.median(cKDTree(xy).query(xy, 2)[0][:, 1])) if xy is not None and len(xy) > 2 else np.nan
    inv.append(f"- {os.path.relpath(f, DD)}: {A.n_obs} x {A.n_vars}, classified as {kind}; obs columns {list(A.obs.columns)[:8]}; obsm {list(A.obsm.keys())}; variable names {list(map(str, A.var_names[:4]))}; "
               f"coordinate range {None if xy is None else (np.round(xy.min(0)).tolist(), np.round(xy.max(0)).tolist())}; nearest-neighbour spacing {nn:.2f}")
    key = re.sub(r"(adata|metabolite|metab|meta|rna|msi|maldi|gene|expr|counts|_+|-+|\.h5ad)", "_", os.path.basename(f).lower()).strip("_")
    key = re.sub(r"_+", "_", key) or os.path.basename(os.path.dirname(f))
    groups.setdefault(key if key else os.path.dirname(f), {})[kind] = f
    if A.isbacked:
        A.file.close()
C.log("\n".join(inv))
pairs = {k: g for k, g in groups.items() if "rna" in g and "metab" in g}
if not pairs:                                           # pair by folder
    bydir = {}
    for f in files:
        A = ad.read_h5ad(f, backed="r"); bydir.setdefault(os.path.dirname(f), {})["metab" if is_metab(A) else "rna"] = f
        A.file.close()
    pairs = {os.path.basename(k): g for k, g in bydir.items() if "rna" in g and "metab" in g}
C.log(f"paired sections: {list(pairs)}")
TOPO = pd.read_csv(os.path.join(P["PROC"], "m20_gem_topology.tsv"), sep="\t").set_index("met_id") if os.path.exists(os.path.join(P["PROC"], "m20_gem_topology.tsv")) else None
GEM = C.load_human_gem(P["REFDB"]); GEM_GENES = set().union(*GEM["met_genes"].values())
gem_ids, _ = make_mapper(GEM, C, TOPO if TOPO is not None else pd.DataFrame(columns=["n_chem_gene", "n_export_gene", "n_internal_transport_gene", "n_total"]))
ION = {pol: (lambda T: T[~T.is_matrix & ((T.adduct != "[M+H-H2O]+") | (T.name == "cholesterol"))])(C.build_ion_table(pol)) for pol in ("positive", "negative")}


def moran_I(Z, xy, k=6):
    _, nn = cKDTree(xy).query(xy, k + 1); nn = nn[:, 1:]
    W = sp.csr_matrix((np.full(nn.size, 1.0 / k), (np.repeat(np.arange(len(xy)), k), nn.ravel())), shape=(len(xy), len(xy)))
    return np.asarray((Z * (W @ Z)).mean(0)).ravel()


def dual_sets(own, uni, dec, mq, n):
    uni = np.setdiff1d(uni, own); key = dec * 5 + mq
    by_k = {k_: uni[key[uni] == k_] for k_ in np.unique(key[uni])}; by_m = {m: uni[mq[uni] == m] for m in range(5)}; out = []
    need = pd.Series(key[own]).value_counts()
    for _ in range(n):
        s = []
        for k_, c in need.items():
            pool = by_k.get(k_, np.array([], int))
            if len(pool) < c:
                pool = by_m.get(k_ % 5, uni)
            s.append(rng.choice(pool, c, replace=len(pool) < c))
        out.append(np.concatenate(s))
    return out


R1, R2 = [], []
for sec, g in pairs.items():
    Ar, Am = ad.read_h5ad(g["rna"]), ad.read_h5ad(g["metab"]); xr, xm = coords(Ar), coords(Am)
    if xr is None:
        C.log(f"  {sec}: RNA without coordinates; skipped"); continue
    sp_r = float(np.median(cKDTree(xr).query(xr, 2)[0][:, 1]))
    # align modalities -> metabolite vector on every RNA spot
    Xm = Am.X.toarray() if sp.issparse(Am.X) else np.asarray(Am.X, float)
    if Am.n_obs == Ar.n_obs and (Am.obs_names.equals(Ar.obs_names) or xm is None or np.allclose(xm, xr)):
        M_spot = Xm; how = "same obs"
    elif xm is not None:
        lo_, hi_ = np.maximum(xr.min(0), xm.min(0)), np.minimum(xr.max(0), xm.max(0)); inter = np.prod(np.clip(hi_ - lo_, 0, None))
        iou = inter / (np.prod(xr.max(0) - xr.min(0)) + np.prod(xm.max(0) - xm.min(0)) - inter)
        if iou < 0.5:
            C.log(f"  {sec}: coordinate boxes of the modalities overlap {iou:.2f} < 0.5 (different coordinate systems); skipped"); continue
        d_, j_ = cKDTree(xr).query(xm); okm = d_ <= 1.5 * sp_r
        cnt = np.bincount(j_[okm], minlength=Ar.n_obs); M_spot = np.zeros((Ar.n_obs, Xm.shape[1]))
        np.add.at(M_spot, j_[okm], Xm[okm]); M_spot = M_spot / np.maximum(cnt, 1)[:, None]; M_spot[cnt == 0] = np.nan; how = f"nearest-neighbour mapping (overlap {iou:.2f})"
    else:
        C.log(f"  {sec}: metabolite data without coordinates and with a different number of rows; skipped"); continue
    # bins
    if a.grid_units:
        R = a.grid_units
    elif Ar.n_obs > 20000:                               # not standard Visium (high-resolution / HD / single cell): bin size for about --target-bins bins (about 200 µm in Visium terms)
        R = sp_r * np.sqrt(Ar.n_obs / a.target_bins)
    else:
        R = a.k_grid * sp_r
    ij = np.floor((xr - xr.min(0)) / R).astype(int); key = ij[:, 1] * 100000 + ij[:, 0]; uk, inv_ = np.unique(key, return_inverse=True)
    nb = len(uk); cnt = np.bincount(inv_, minlength=nb); okb = cnt >= 2
    Xr = Ar.X.tocsr() if sp.issparse(Ar.X) else sp.csr_matrix(np.asarray(Ar.X, float)); isint = np.allclose(Xr.data[:1000], np.round(Xr.data[:1000]))
    Bm = sp.csr_matrix((np.ones(len(inv_)), (inv_, np.arange(len(inv_)))), shape=(nb, len(inv_)))
    CBn = (Bm @ Xr).tocsr()
    if isint:
        tot = np.asarray(CBn.sum(1)).ravel(); LGb = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CBn).toarray())
    else:
        LGb = (sp.diags(1 / np.maximum(cnt, 1)) @ CBn).toarray()
    Mfin = np.where(np.isfinite(M_spot), M_spot, 0); Mcnt = np.isfinite(M_spot).astype(float)
    Mb = (Bm @ Mfin) / np.maximum(Bm @ Mcnt, 1)
    okb &= (Bm @ Mcnt).sum(1) > 0
    xy = np.column_stack([uk % 100000, uk // 100000])[okb] * R; LG = LGb[okb]; MB = Mb[okb]
    if np.nanmin(MB) >= 0 and np.nanmax(MB) > 50:
        MB = np.log1p(MB)
    gn = np.asarray(Ar.var_names).astype(str); gnu = np.char.upper(gn); keepg = (LG > 0).mean(0) >= 0.01
    LG, gnu = LG[:, keepg], gnu[keepg]; pos = {g: i for i, g in enumerate(gnu)}
    mz = mz_of(Am); detm = (MB > np.nanmin(MB, axis=0)).mean(0)
    Zg = np.nan_to_num(C.zcols(LG)); I_ = moran_I(Zg, xy); dec = np.minimum((pd.Series(LG.mean(0)).rank(pct=True).values * 10).astype(int), 9)
    mq = np.minimum((pd.Series(I_).rank(pct=True).values * 5).astype(int), 4); U_all = np.arange(LG.shape[1]); U_gem = np.array([i for i, g in enumerate(gnu) if g in GEM_GENES], int)
    # polarity
    hits = {}
    for pol, T in ION.items():
        hits[pol] = len({T.name.values[k] for m_ in mz[np.isfinite(mz)] for k in np.where(np.abs(T.mz.values - m_) / m_ * 1e6 <= 3)[0]})
    pol = a.polarity if a.polarity != "auto" else max(hits, key=hits.get)
    C.log(f"  {sec}: RNA {Ar.n_obs} spots x {Ar.n_vars} genes (spacing {sp_r:.1f} units, {'counts' if isint else 'normalized'}); metabolites {Am.n_obs} x {Am.n_vars} ({how}); "
          f"bins of {R:.1f} units -> {okb.sum()} bins; polarity {pol} ({hits['positive']} positive / {hits['negative']} negative metabolites matched)")
    # T1 NAA
    js = []
    for ad_, t in NAA[pol].items():
        d_ = np.abs(mz - t) / t * 1e6
        if np.nanmin(d_) <= 5:
            j = int(np.nanargmin(d_))
            if detm[j] >= 0.05:
                js.append(j)
    if js:
        y = C.zcols(C.zcols(MB[:, js]).mean(1)[:, None])[:, 0]; row = dict(section=sec, n_bins=len(y), naa_ions=",".join(f"{mz[j]:.4f}" for j in js))
        for side, genes in (("neuronal", NEUR_SIDE), ("oligodendroglial", OLIG_SIDE)):
            own = np.array([pos[g] for g in genes if g in pos], int)
            if len(own) == 0:
                continue
            sc = Zg[:, own].mean(1); r = float(np.corrcoef(sc, y)[0, 1])
            nul = np.array([np.corrcoef(Zg[:, s_].mean(1), y)[0, 1] for s_ in dual_sets(own, U_all, dec, mq, a.n_rand)]); nul = nul[np.isfinite(nul)]
            row.update({f"{side}_n_genes": len(own), f"{side}_r": r, f"{side}_p_dual": float((np.sum(nul >= r) + 1) / (len(nul) + 1))})
        Zm = np.nan_to_num(C.zcols(MB[:, detm >= 0.05])); own = np.array([pos[g] for g in NEUR_SIDE if g in pos], int)
        if len(own):
            rall = Zm.T @ C.zcols(Zg[:, own].mean(1)[:, None])[:, 0] / len(y); row["NAA_rank_pct_neuronal"] = float((rall < row.get("neuronal_r", np.nan)).mean())
        R1.append(row); C.log("     NAA: " + json.dumps({k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in row.items()}, ensure_ascii=False))
    else:
        C.log(f"     NAA ({pol}) not detected (<= 5 ppm and detected >= 5%)")
    # T2 own GEM pathway
    T = ION[pol]; feats = {}
    for j, m_ in enumerate(mz):
        if not np.isfinite(m_) or detm[j] < 0.10:
            continue
        h = T[np.abs(T.mz.values - m_) / m_ * 1e6 <= 5]
        if len(h) and h.name.nunique() == 1:
            feats.setdefault(h.name.iloc[0], []).append(j)
    folds = C.spatial_folds(xy, 5 * R, seed=1)
    for nm, js_ in feats.items():
        ids = gem_ids(nm)
        if not ids:
            continue
        own_g = set().union(*[GEM["met_genes"].get(i, set()) for i in ids]); own = np.array(sorted(pos[g] for g in own_g if g in pos), int)
        if not (2 <= len(own) <= 150):
            continue
        y = C.zcols(C.zcols(MB[:, js_]).mean(1)[:, None])[:, 0]
        sets = [own] + dual_sets(own, U_gem, dec, mq, a.n_null); r_ = batch_cv_r2(batch_X(Zg, sets, 3), y, folds); nul = r_[1:]
        R2.append(dict(section=sec, metabolite=nm, n_genes=len(own), R2_own=float(r_[0]), null_med=float(np.median(nul)), pctl=float(np.mean(nul < r_[0])), p=float((np.sum(nul >= r_[0]) + 1) / (len(nul) + 1))))
        C.log(f"     own GEM pathway: {sum(1 for r in R2 if r['section'] == sec)} metabolites")
A1, A2 = pd.DataFrame(R1), pd.DataFrame(R2)
A1.to_csv(os.path.join(TAB, "m23_naa.tsv"), sep="\t", index=False); A2.to_csv(os.path.join(TAB, "m23_gem.tsv"), sep="\t", index=False)
v = dict(n_sections=len(pairs), data_dir=DD)
if len(A1) and "neuronal_r" in A1 and "oligodendroglial_r" in A1:
    v["D64a_NAA_autonomy_external"] = bool((A1.neuronal_r > A1.oligodendroglial_r).all() and (A1.neuronal_p_dual < 0.05).mean() >= 0.5)
    v["NAA"] = A1.round(4).to_dict("records")
else:
    v["D64a_NAA_autonomy_external"] = "not testable"
if len(A2):
    k = int((A2.p < 0.05).sum()); bp = float(binomtest(k, len(A2), 0.05, alternative="greater").pvalue); fr = k / len(A2)
    v["D64b_sparse_specificity_external"] = bool(bp < 0.05 and fr < 0.30); v["gem"] = dict(n_tests=len(A2), frac_sig=fr, binom_p=bp, top=A2.sort_values("p").head(8)[["section", "metabolite", "pctl", "p"]].round(3).to_dict("records"))
else:
    v["D64b_sparse_specificity_external"] = "not testable"
C.save_json(v, os.path.join(TAB, "m23_verdict.json")); C.log("decision: " + json.dumps(v, ensure_ascii=False, default=str)[:2500])
if len(A1) or len(A2):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    if len(A1):
        x = np.arange(len(A1)); axs[0].bar(x - .2, A1.get("neuronal_r", 0), .4, color="#de2d26", label="neuronal (NAT8L + neuron genes)"); axs[0].bar(x + .2, A1.get("oligodendroglial_r", 0), .4, color="#3182bd", label="oligodendroglial (ASPA + myelin)")
        axs[0].set_xticks(x); axs[0].set_xticklabels(A1.section, fontsize=6); axs[0].axhline(0, c="k", lw=.5); axs[0].legend(fontsize=6, frameon=False); axs[0].set_ylabel("r(NAA, gene score)")
        axs[0].set_title("External dataset (mouse hippocampus): NAA follows its synthesising side?", fontsize=7)
    if len(A2):
        axs[1].hist(A2.pctl, bins=20, color="#3182bd"); axs[1].axvline(0.95, c="k", lw=.5, ls="--"); axs[1].set_xlabel("own-pathway percentile (dual-matched null)")
        axs[1].set_title(f"External: {100 * v['gem']['frac_sig']:.0f}% P<0.05 (binomial P = {v['gem']['binom_p']:.2g})", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m23_external.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
