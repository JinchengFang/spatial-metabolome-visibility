#!/usr/bin/env python3

import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import binomtest, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import p2_common as C, p2_loaders as L
from m20_helpers import make_mapper, batch_X, batch_cv_r2
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200); ap.add_argument("--n-rand", type=int, default=999)
ap.add_argument("--n-null", type=int, default=499); ap.add_argument("--n-pc", type=int, default=3); ap.add_argument("--step", default="Step17")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(1705); R = a.res
GEM = C.load_human_gem(P["REFDB"]); GEM_GENES = set().union(*GEM["met_genes"].values())
TOPO = pd.read_csv(os.path.join(P["PROC"], "m20_gem_topology.tsv"), sep="\t").set_index("met_id"); gem_ids, topo_of = make_mapper(GEM, C, TOPO)
TARGET = {"dopamine": ["TH", "DDC", "SLC6A3", "SLC18A2"], "serotonin": ["TPH2", "SLC6A4", "FEV"], "GABA": ["GAD1", "GAD2", "SLC32A1"]}
NAA_SIDE = ["NAT8L"] + C.PROGRAMS["Neuron_cortex"].split(); NAA_MZ = {"DHB": [176.0553, 198.0373, 214.0113], "9AA": [174.0408, 210.0175]}


def moran_I(Z, xy, k=6):
    """Moran's I of every (standardized) column with row-standardized k-nearest-neighbour weights."""
    _, nn = cKDTree(xy).query(xy, k + 1); nn = nn[:, 1:]
    W = sp.csr_matrix((np.full(nn.size, 1.0 / k), (np.repeat(np.arange(len(xy)), k), nn.ravel())), shape=(len(xy), len(xy)))
    return np.asarray((Z * (W @ Z)).mean(0)).ravel()


def strata(LG, xy):
    mexp = LG.mean(0); dec = np.minimum((pd.Series(mexp).rank(pct=True).values * 10).astype(int), 9)
    I = moran_I(np.nan_to_num(C.zcols(LG)), xy); mq = np.minimum((pd.Series(I).rank(pct=True).values * 5).astype(int), 4)
    return dec, mq, I


def dual_sets(own, uni, dec, mq, n):
    """Random set of the size of own: each gene replaced by a gene of the same (expression decile, Moran quintile); falls back to the same Moran quintile, then the same expression decile."""
    uni = np.setdiff1d(uni, own); key = dec * 5 + mq
    by_k = {k_: uni[key[uni] == k_] for k_ in np.unique(key[uni])}; by_m = {m: uni[mq[uni] == m] for m in range(5)}; by_d = {d: uni[dec[uni] == d] for d in range(10)}
    need = pd.Series(key[own]).value_counts(); out = []
    for _ in range(n):
        s = []
        for k_, c in need.items():
            pool = by_k.get(k_, np.array([], int))
            if len(pool) < c:
                pool = by_m.get(k_ % 5, np.array([], int))
            if len(pool) < c:
                pool = by_d.get(k_ // 5, uni)
            s.append(rng.choice(pool, c, replace=len(pool) < c))
        out.append(np.concatenate(s))
    return out


def corr_test(y, Zg, own, uni, dec, mq, n):
    sc = Zg[:, own].mean(1)
    if sc.std() < 1e-9:
        return np.nan, np.nan, np.nan
    r = float(np.corrcoef(sc, y)[0, 1]); nul = np.array([np.corrcoef(Zg[:, s_].mean(1), y)[0, 1] for s_ in dual_sets(own, uni, dec, mq, n)]); nul = nul[np.isfinite(nul)]
    return r, float((np.sum(nul >= r) + 1) / (len(nul) + 1)), float(np.median(nul))


import anndata as ad
cfg = L.Config(P, min_px_half=0, min_bins=40); ION = {pol: C.build_ion_table(pol) for pol in ("positive", "negative")}
T1, T3 = [], []
for S in L.load_all(cfg, ["SMA"]):
    mtx = str(S.get("matrix", "")).upper().replace("-", "")
    kind = "FMP10" if "FMP" in mtx else ("DHB" if "DHB" in mtx else ("9AA" if "9AA" in mtx else None))
    if kind is None:
        continue
    G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    Y0, cnt = L.pixel_means(S, G, None, (0.0, 0.0)); ok = cnt >= 1; Y = Y0[ok]; xy = G["xy"][ok]; det = (np.expm1(Y) > 0).mean(0)
    var = ad.read_h5ad(os.path.join(P["PROC"], "sma", f"{S['msi_section']}_paired.h5ad"), backed="r").var; mz, lo, hi = var.mz.values, var.win_lo.values, var.win_hi.values
    CB = G["CB"][np.where(ok)[0]].tocsc(); tot = np.asarray(CB.sum(1)).ravel(); gn = np.asarray(S["genes"]).astype(str)
    keep = np.where(np.asarray((CB > 0).mean(0)).ravel() >= 0.01)[0]
    keep = np.union1d(keep, [i for i, g in enumerate(gn) if g in set(sum(TARGET.values(), []) + NAA_SIDE)]).astype(int)
    LG = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CB[:, keep]).toarray()).astype(np.float32); gk = gn[keep]; gidx = {g: i for i, g in enumerate(gk)}
    dec, mq, I = strata(LG, xy); Zg = np.nan_to_num(C.zcols(LG)); U_all = np.arange(len(gk)); U_gem = np.array([i for i, g in enumerate(gk) if g in GEM_GENES], int)
    tol = lambda j, t: abs(mz[j] - t) / t * 1e6 <= max(5, (hi[j] - lo[j]) / 2 / t * 1e6 + 2)
    # ---- T1 / T2 targeted tests
    tests = []
    if kind == "FMP10":
        for mol, genes in TARGET.items():
            M, ns, _ = C.NEURO_PANEL[mol]; js = []
            for _, t in C.fmp10_ions(M, ns):
                j = int(np.argmin(np.abs(mz - t)))
                if tol(j, t) and det[j] >= 0.05:
                    js.append(j)
            tests.append((mol, sorted(set(js)), genes))
    else:
        js = [int(np.argmin(np.abs(mz - t))) for t in NAA_MZ[kind]]; js = [j for j, t in zip(js, NAA_MZ[kind]) if tol(j, t) and det[j] >= 0.05]
        tests.append(("N-acetylaspartate", js, NAA_SIDE))
    for mol, js, genes in tests:
        own = np.array([gidx[g] for g in genes if g in gidx], int)
        if not js or not len(own):
            continue
        y = C.zcols(C.zcols(Y[:, js]).mean(1)[:, None])[:, 0]
        r, p_dual, med = corr_test(y, Zg, own, U_all, dec, mq, a.n_rand)
        r2, p_expr, _ = corr_test(y, Zg, own, U_all, dec, np.zeros_like(mq), a.n_rand)          # expression-matched only (= the null of Steps 13-15), for comparison
        T1.append(dict(sample=S["sample"], matrix=kind, brain_area=S.get("brain_area"), molecule=mol, n_ions=len(js), n_genes=len(own), r=r, p_expr_matched=p_expr,
                       p_dual_matched=p_dual, null_median_dual=med, geneset_moran_pctl=float(np.mean(I[None, :] < I[own][:, None])), geneset_moran=float(I[own].mean())))
    # ---- T3 own GEM pathways (annotation and filters as in M20-01)
    feats = {}
    if kind == "FMP10":
        A_ = C.annotate_fmp10(mz, lo, hi)
        for r_ in (A_[A_.n_molecules_on_peak == 1].itertuples() if len(A_) else []):
            feats.setdefault(r_.molecule, set()).add(int(r_.peak_index))
    else:
        Tb = ION["positive" if kind == "DHB" else "negative"]; Tb = Tb[~Tb.is_matrix & ((Tb.adduct != "[M+H-H2O]+") | (Tb.name == "cholesterol"))]
        for j, m_ in enumerate(mz):
            h = Tb[np.abs(Tb.mz.values - m_) / m_ * 1e6 <= max(5.0, (hi[j] - lo[j]) / 2 / m_ * 1e6 + 2)]
            if len(h) and h.name.nunique() == 1:
                feats.setdefault(h.name.iloc[0], set()).add(j)
    folds = C.spatial_folds(xy, 1000, seed=R)
    for nm, js in feats.items():
        js = [j for j in js if det[j] >= 0.10]; ids = gem_ids(nm)
        if not js or not ids:
            continue
        own_g = set().union(*[GEM["met_genes"].get(i, set()) for i in ids]); own = np.array(sorted(gidx[g] for g in own_g if g in gidx), int)
        if not (2 <= len(own) <= 150):
            continue
        y = C.zcols(C.zcols(Y[:, js]).mean(1)[:, None])[:, 0]
        sets = [own] + dual_sets(own, U_gem, dec, mq, a.n_null); r_ = batch_cv_r2(batch_X(Zg, sets, a.n_pc), y, folds); nul = r_[1:]
        T3.append(dict(sample=S["sample"], matrix=kind, brain_area=S.get("brain_area"), metabolite=nm, n_genes=len(own), R2_own=float(r_[0]), null_med=float(np.median(nul)),
                       pctl=float(np.mean(nul < r_[0])), p=float((np.sum(nul >= r_[0]) + 1) / (len(nul) + 1)), geneset_moran_pctl=float(np.mean(I[None, :] < I[own][:, None])),
                       export_share=topo_of(ids)["export_share"]))
    C.log(f"  {S['sample']} ({kind}): {sum(1 for t in T1 if t['sample'] == S['sample'])} targeted tests; {sum(1 for t in T3 if t['sample'] == S['sample'])} GEM pathway tests")
A1, A3 = pd.DataFrame(T1), pd.DataFrame(T3)
A1.to_csv(os.path.join(TAB, "m20_strict_targeted.tsv"), sep="\t", index=False); A3.to_csv(os.path.join(TAB, "m20_strict_gem.tsv"), sep="\t", index=False)
C.log("targeted tests (expression-matched vs expression x spatial-structure matched):\n" + A1.round(3).to_string(index=False))
v = {}
da = A1[A1.molecule == "dopamine"]; sn, st = da[da.brain_area == "Substantia Nigra"], da[da.brain_area == "Striatum"]
if len(sn) and len(st):
    v["D54_strict"] = bool((sn.p_dual_matched < 0.05).mean() >= 2 / 3 and (st.p_dual_matched < 0.05).mean() <= 1 / 3)
    v["dopamine"] = dict(SN_sig_dual=f"{int((sn.p_dual_matched < 0.05).sum())}/{len(sn)}", STR_sig_dual=f"{int((st.p_dual_matched < 0.05).sum())}/{len(st)}",
                         SN_sig_expr=f"{int((sn.p_expr_matched < 0.05).sum())}/{len(sn)}", SN_geneset_moran_pctl=float(sn.geneset_moran_pctl.median()))
na = A1[A1.molecule == "N-acetylaspartate"]
if len(na):
    v["D58a_strict"] = bool((na.p_dual_matched < 0.05).mean() >= 2 / 3); v["NAA"] = dict(sig_dual=f"{int((na.p_dual_matched < 0.05).sum())}/{len(na)}", r=na.r.round(3).tolist())
ht = A1[A1.molecule == "serotonin"]
if len(ht):
    v["ctrl_5HT_still_null"] = bool((ht.p_dual_matched < 0.05).mean() <= 1 / 3)
if len(A3):
    k = int((A3.p < 0.05).sum()); v["D59a_strict"] = bool(binomtest(k, len(A3), 0.05, alternative="greater").pvalue < 0.05)
    g = A3.groupby("metabolite").agg(pctl=("pctl", "mean"), x=("export_share", "first")).dropna()
    v["gem"] = dict(n_tests=len(A3), frac_sig=float(k / len(A3)), binom_p=float(binomtest(k, len(A3), 0.05, alternative="greater").pvalue),
                    rho_export=float(spearmanr(g.x, g.pctl).correlation) if len(g) >= 10 else None, n_metabolites=len(g))
C.save_json(v, os.path.join(TAB, "m20_strict_verdict.json")); C.log("decision: " + json.dumps(v, ensure_ascii=False, default=str))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
if len(A1):
    lab = A1["molecule"].str.replace("N-acetylaspartate", "NAA") + "\n" + A1["sample"].str.extract(r"(\d+[-_][a-d]\d)")[0].fillna("") + " " + A1.brain_area.fillna("").str.replace("Substantia Nigra", "SN").str.replace("Striatum", "STR")
    x = np.arange(len(A1))
    axs[0].scatter(x - .12, -np.log10(A1.p_expr_matched), c="#bdbdbd", s=16, label="expression-matched null (Steps 13–15)")
    axs[0].scatter(x + .12, -np.log10(A1.p_dual_matched), c="#de2d26", s=16, label="expression × spatial-structure matched")
    axs[0].axhline(-np.log10(0.05), c="k", lw=.5, ls="--"); axs[0].set_xticks(x); axs[0].set_xticklabels(lab, rotation=90, fontsize=4.5); axs[0].set_ylabel("−log10 P"); axs[0].legend(fontsize=5, frameon=False)
    axs[0].set_title("Targeted tests under a stricter null", fontsize=7)
if len(A3):
    axs[1].hist(A3.pctl, bins=20, color="#3182bd"); axs[1].axvline(0.95, c="k", lw=.5, ls="--"); axs[1].set_xlabel("own-pathway percentile (dual-matched null)")
    axs[1].set_title(f"GEM own-pathway: {100 * v['gem']['frac_sig']:.0f}% P<0.05 (binomial P = {v['gem']['binom_p']:.2g})", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m20_strict_null.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
