#!/usr/bin/env python3

import argparse, glob, json, os, re, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu, binomtest, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import p2_common as C, p2_loaders as L
from m20_gem_topology import gem_topology
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200); ap.add_argument("--n-null", type=int, default=199)
ap.add_argument("--n-pc", type=int, default=3); ap.add_argument("--min-genes", type=int, default=2); ap.add_argument("--max-set", type=int, default=150)
ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--n-perm", type=int, default=9999); ap.add_argument("--step", default="Step16")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
rng = np.random.default_rng(16); R = a.res
GEM = C.load_human_gem(P["REFDB"]); GEM_GENES = set().union(*GEM["met_genes"].values())
TOPO = gem_topology(os.path.join(P["REFDB"], "Human-GEM", "Human-GEM.yml"), os.path.join(P["REFDB"], "Human-GEM", "genes.tsv"),
                    cache=os.path.join(P["PROC"], "m20_gem_topology.tsv")).set_index("met_id")
C.log(f"Human-GEM topology: {len(TOPO)} metabolites; median export_share {TOPO.export_share.median():.2f}")
SYN = {"norepinephrine": ["noradrenaline"], "DOPAC": ["3,4-dihydroxyphenylacetate"], "homovanillic acid": ["homovanillate"], "5-HIAA": ["5-hydroxyindoleacetate"],
       "glutamate": ["l-glutamate", "glutamate"], "glutamine": ["l-glutamine", "glutamine"], "tyrosine": ["l-tyrosine", "tyrosine"], "alanine": ["l-alanine", "alanine"],
       "3-methoxytyramine": ["3-methoxytyramine"], "acetylcholine": ["acetylcholine"]}
RELEASED = {"dopamine", "norepinephrine", "serotonin", "GABA", "glutamate", "glycine", "acetylcholine", "histamine", "adenosine", "N-acetylaspartylglutamate", "ATP"}


def gem_ids(nm):
    nm = re.sub(r"\s*\[[^\]]*\][+-]?\s*$", "", str(nm).strip())
    cands = SYN.get(nm, []) + C.GEM_SYNONYM.get(nm, []) + [nm.lower(), "l-" + nm.lower()]
    ids = set()
    for c in cands:
        ids |= GEM["name2ids"].get(c.lower(), set())
    return {i for i in ids if GEM["n_rxn"].get(i, 0) <= 400}


def topo_of(ids):
    t = TOPO.reindex(sorted(ids)).dropna(how="all")
    if not len(t):
        return dict(n_chem_gene=np.nan, n_export_gene=np.nan, export_share=np.nan)
    ch, ex = float(t.n_chem_gene.sum()), float(t.n_export_gene.sum())
    return dict(n_chem_gene=ch, n_export_gene=ex, export_share=ex / max(ch + ex, 1))


# ---------- batched specificity test (same algorithm as M9-02) ----------
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])


def batch_X(Gz, sets, k):
    out = []
    for c0 in range(0, len(sets), 50):
        A = np.stack([Gz[:, s_] for s_ in sets[c0:c0 + 50]])
        if A.shape[2] > k:
            U, S_, _ = np.linalg.svd(A, full_matrices=False); A = U[:, :, :k] * S_[:, None, :k]
        out.append(A)
    return np.concatenate(out)


def batch_cv_r2(Xb, y, folds):
    N, n, p = Xb.shape; pred = np.empty((N, n)); I_ = np.eye(p)
    for f_ in np.unique(folds):
        te = folds == f_; tr = ~te
        Xt = Xb[:, tr]; mu = Xt.mean(1, keepdims=True); sd = Xt.std(1, keepdims=True) + 1e-9
        Xt = (Xt - mu) / sd; Xs = (Xb[:, te] - mu) / sd; yt = y[tr] - y[tr].mean()
        XtX = np.matmul(Xt.transpose(0, 2, 1), Xt); Xty = np.matmul(Xt.transpose(0, 2, 1), yt[None, :, None])[:, :, 0]
        best = np.full(N, np.inf); bb = np.zeros((N, p))
        for al in ALPHAS:
            Minv = np.linalg.inv(XtX + al * I_); b = np.matmul(Minv, Xty[:, :, None])[:, :, 0]
            fit = np.matmul(Xt, b[:, :, None])[:, :, 0]; h = (np.matmul(Xt, Minv) * Xt).sum(2)
            loo = (((yt[None, :] - fit) / (1 - h)) ** 2).mean(1); m = loo < best; best[m] = loo[m]; bb[m] = b[m]
        pred[:, te] = np.matmul(Xs, bb[:, :, None])[:, :, 0] + y[tr].mean()
    return 1 - ((y[None, :] - pred) ** 2).sum(1) / ((y - y.mean()) ** 2).sum()


def matched_sets(own, uni, dec, n):
    uni = np.setdiff1d(uni, own); by = {d: uni[dec[uni] == d] for d in np.unique(dec[uni])}; need = pd.Series(dec[own]).value_counts(); out = []
    for _ in range(n):
        s = []
        for d, k in need.items():
            pool = by.get(d, np.array([], int)); pool = pool if len(pool) >= k else uni
            s.append(rng.choice(pool, k, replace=False))
        out.append(np.concatenate(s))
    return out


# ---------- A mouse, same section ----------
import anndata as ad
cfg = L.Config(P, min_px_half=0, min_bins=40); rows = []
ION = {pol: C.build_ion_table(pol) for pol in ("positive", "negative")}
for S in L.load_all(cfg, ["SMA"]):
    mtx = str(S.get("matrix", "")).upper().replace("-", "")
    kind = "FMP10" if "FMP" in mtx else ("DHB" if "DHB" in mtx else ("9AA" if "9AA" in mtx else None))
    if kind is None:
        continue
    G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    Y0, cnt = L.pixel_means(S, G, None, (0.0, 0.0)); ok = cnt >= 1; Y = Y0[ok]; xy = G["xy"][ok]
    var = ad.read_h5ad(os.path.join(P["PROC"], "sma", f"{S['msi_section']}_paired.h5ad"), backed="r").var
    mz, lo, hi = var.mz.values, var.win_lo.values, var.win_hi.values; det = (np.expm1(Y) > 0).mean(0)
    feats = {}
    if kind == "FMP10":
        A_ = C.annotate_fmp10(mz, lo, hi)
        if len(A_):
            for r_ in A_[A_.n_molecules_on_peak == 1].itertuples():
                feats.setdefault(r_.molecule, set()).add(int(r_.peak_index))
    else:
        T = ION["positive" if kind == "DHB" else "negative"]; T = T[~T.is_matrix & ((T.adduct != "[M+H-H2O]+") | (T.name == "cholesterol"))]
        for j, m_ in enumerate(mz):
            tol = max(5.0, (hi[j] - lo[j]) / 2 / m_ * 1e6 + 2); h = T[np.abs(T.mz.values - m_) / m_ * 1e6 <= tol]
            if len(h) and h.name.nunique() == 1:
                feats.setdefault(h.name.iloc[0], set()).add(j)
    CB = G["CB"][np.where(ok)[0]].tocsc(); tot = np.asarray(CB.sum(1)).ravel(); gn = np.asarray(S["genes"]).astype(str)
    keep = np.where(np.asarray((CB > 0).mean(0)).ravel() >= 0.01)[0]
    LG = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CB[:, keep]).toarray()).astype(np.float32); gk = gn[keep]; gidx = {g: i for i, g in enumerate(gk)}
    dec = np.minimum((pd.Series(LG.mean(0)).rank(pct=True).values * 10).astype(int), 9)
    U = np.array([i for i, g in enumerate(gk) if g in GEM_GENES], int); folds = C.spatial_folds(xy, 1000, seed=R); T2 = C.poly2(xy)
    Gf = np.nan_to_num(C.zcols(LG)); Gd = np.nan_to_num(C.zcols(C.detrend(LG, T2))); n_ok = 0
    for nm, js in feats.items():
        js = [j for j in js if det[j] >= a.min_det]
        if not js:
            continue
        ids = gem_ids(nm)
        if not ids:
            continue
        own_g = set().union(*[GEM["met_genes"].get(i, set()) for i in ids]); own = np.array(sorted(gidx[g] for g in own_g if g in gidx), int)
        if not (a.min_genes <= len(own) <= a.max_set):
            continue
        y = C.zcols(C.zcols(Y[:, js]).mean(1)[:, None])[:, 0]
        for level, Gz, yy in (("full", Gf, y), ("detrended", Gd, C.zcols(C.detrend(y[:, None], T2))[:, 0])):
            sets = [own] + matched_sets(own, U, dec, a.n_null); r_ = batch_cv_r2(batch_X(Gz, sets, a.n_pc), yy, folds); nul = r_[1:]
            rows.append(dict(dataset="SMA-mouse", sample=S["sample"], matrix=kind, brain_area=S.get("brain_area"), level=level, metabolite=nm, n_ions=len(js),
                             detection=float(np.mean(det[js])), n_genes=len(own), genes=";".join(gk[own][:20]), R2_own=float(r_[0]), null_med=float(np.median(nul)),
                             null_sd=float(nul.std()), pctl=float(np.mean(nul < r_[0])), p=float((np.sum(nul >= r_[0]) + 1) / (len(nul) + 1)),
                             z=float((r_[0] - nul.mean()) / (nul.std() + 1e-9)), **topo_of(ids)))
        n_ok += 1
    C.log(f"  {S['sample']} ({kind}, {S.get('brain_area')}): {len(feats)} uniquely annotated metabolites -> {n_ok} testable (detected >= {a.min_det:.0%}, {a.min_genes}-{a.max_set} GEM genes)")
SM = pd.DataFrame(rows)
if not len(SM):
    raise SystemExit("no testable mouse metabolites")
SM.to_csv(os.path.join(TAB, "m20_sma_specificity.tsv"), sep="\t", index=False)
C.log("mouse (full):\n" + SM[SM.level == "full"].sort_values("pctl", ascending=False)[["sample", "matrix", "metabolite", "n_genes", "R2_own", "null_med", "pctl", "p", "export_share"]].round(3).to_string(index=False))

# ---------- B human tumours (M9-02 results) ----------
HU = None
cands = sorted(glob.glob(os.path.join(P["P2"], "05_tables", "Step*", "m9_gem_specificity.tsv")), key=lambda f: int(re.search(r"Step(\d+)", f).group(1)))
if cands:
    HU = pd.read_csv(cands[-1], sep="\t"); HU = HU[(HU.level == "full") & (HU.res_um == R)] if "level" in HU else HU
    tp = []
    for r_ in HU.itertuples():
        ids = set().union(*[gem_ids(x) for x in str(r_.gem_match).split("|")]); tp.append(topo_of(ids))
    HU = pd.concat([HU.reset_index(drop=True), pd.DataFrame(tp)], axis=1); HU["metabolite"] = HU.gem_match.astype(str)
    C.log(f"human tumours M9-02 ({os.path.relpath(cands[-1], P['P2'])}): {len(HU)} metabolite x sample")


# ---------- C metabolite level ----------
def spectrum(D, pcol):
    g = D.groupby("metabolite").agg(pctl=(pcol, "mean"), n_sections=(pcol, "size"), frac_sig=("p" if "p" in D else "p_gem", lambda s: float((s < 0.05).mean())),
                                    export_share=("export_share", "first"), n_chem_gene=("n_chem_gene", "first"), n_export_gene=("n_export_gene", "first"),
                                    n_genes=("n_genes" if "n_genes" in D else "n_genes_expr", "median")).reset_index()
    g["released"] = g.metabolite.isin(RELEASED); return g.dropna(subset=["export_share"])


def perm_spearman(x, y, n):
    rho = spearmanr(x, y).correlation; rx = rankdata(x); ry = rankdata(y)
    nul = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return float(rho), float((np.sum(nul <= rho) + 1) / (n + 1))


def partial_spearman(x, y, zc):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(zc); Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


v = {}; specs = {}
for level in ("full", "detrended"):
    s = SM[SM.level == level]; sp_ = spectrum(s, "pctl"); specs[level] = sp_
    out = dict(n_tests=len(s), frac_sig=float((s.p < 0.05).mean()), binom_p=float(binomtest(int((s.p < 0.05).sum()), len(s), 0.05, alternative="greater").pvalue),
               n_metabolites=len(sp_))
    if len(sp_) >= 5:
        out["rho"], out["p_perm"] = perm_spearman(sp_.export_share.values, sp_.pctl.values, a.n_perm)
        out["partial_rho_given_setsize"] = partial_spearman(sp_.export_share.values, sp_.pctl.values, sp_.n_genes.values)
        if sp_.released.sum() >= 2 and (~sp_.released).sum() >= 2:
            out["released_mean_pctl"] = float(sp_[sp_.released].pctl.mean()); out["retained_mean_pctl"] = float(sp_[~sp_.released].pctl.mean())
            out["mwu_p"] = float(mannwhitneyu(sp_[sp_.released].pctl, sp_[~sp_.released].pctl, alternative="less").pvalue)
    v[f"mouse_{level}"] = out
f = v["mouse_full"]
v["D59a_specificity_exists_same_section"] = bool(f["binom_p"] < 0.05)
v["D59b_export_predicts_low_autonomy"] = bool(f.get("n_metabolites", 0) >= 10 and f.get("rho", 0) < 0 and f.get("p_perm", 1) < 0.05)
v["D59c_released_lower"] = bool(f.get("mwu_p", 1) < 0.05) if "mwu_p" in f else None
if HU is not None and len(HU):
    hs = spectrum(HU.assign(p=HU.p_gem), "pctl_gem"); specs["human"] = hs
    if len(hs) >= 5:
        rho, pp = perm_spearman(hs.export_share.values, hs.pctl.values, a.n_perm)
        v["human"] = dict(n_metabolites=len(hs), rho=rho, p_perm=pp, frac_sig=float((HU.p_gem < 0.05).mean())); v["D59d_human_same_sign"] = bool(rho < 0)
anch = specs["full"].set_index("metabolite")
v["anchors"] = {k: dict(pctl=round(float(anch.loc[k, "pctl"]), 3), export_share=round(float(anch.loc[k, "export_share"]), 2)) for k in ("N-acetylaspartate", "dopamine", "GABA", "glutamate", "choline", "cholesterol", "spermidine") if k in anch.index}
da = SM[(SM.level == "full") & (SM.metabolite == "dopamine")]
if len(da):
    v["dopamine_by_area"] = da.groupby("brain_area").pctl.mean().round(3).to_dict()
SPEC = pd.concat([d.assign(cohort=k) for k, d in specs.items()], ignore_index=True)
SPEC.to_csv(os.path.join(TAB, "m20_metabolite_spectrum.tsv"), sep="\t", index=False)
C.save_json(v, os.path.join(TAB, "m20_verdict.json")); C.log("metabolite level (full):\n" + specs["full"].sort_values("pctl", ascending=False).round(3).to_string(index=False) + "\ndecision: " + json.dumps(v, ensure_ascii=False, default=str))

# ---------- figure ----------
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax_, key, title in ((axs[0], "full", "Mouse same-section (SMA)"), (axs[2], "human", "Human tumours (adjacent sections; Step 7 M9-02)")):
    if key not in specs:
        ax_.axis("off"); continue
    d = specs[key]; c_ = ["#de2d26" if r else "#3182bd" for r in d.released]
    ax_.scatter(d.export_share, d.pctl, s=12 + 6 * d.n_sections, c=c_, alpha=.8, edgecolor="k", lw=.3)
    for r_ in d.itertuples():
        if r_.pctl > 0.85 or r_.released or r_.metabolite in ("N-acetylaspartate", "cholesterol", "spermidine", "choline"):
            ax_.text(r_.export_share, r_.pctl, " " + r_.metabolite[:18], fontsize=5)
    vv = v["mouse_full"] if key == "full" else v.get("human", {})
    ax_.axhline(0.5, c="grey", lw=.5, ls="--"); ax_.set_xlabel("export share in Human-GEM (plasma-membrane transport / (transport + chemistry))")
    ax_.set_ylabel("own-pathway specificity (percentile vs matched GEM gene sets)")
    ax_.set_title(f"{title}: ρ = {vv.get('rho', np.nan):+.2f}, P = {vv.get('p_perm', np.nan):.3f}", fontsize=7)
d = specs["full"]
axs[1].boxplot([d[~d.released].pctl, d[d.released].pctl], widths=.5); axs[1].set_xticks([1, 2]); axs[1].set_xticklabels(["retained / other", "released transmitters"])
for i, grp in enumerate([d[~d.released], d[d.released]]):
    axs[1].scatter(np.full(len(grp), i + 1) + rng.uniform(-.12, .12, len(grp)), grp.pctl, s=8, c=["#3182bd", "#de2d26"][i])
axs[1].axhline(0.5, c="grey", lw=.5, ls="--"); axs[1].set_ylabel("own-pathway specificity (percentile)")
axs[1].set_title(f"Mouse: released vs other (one-sided MWU P = {v['mouse_full'].get('mwu_p', np.nan):.3f})", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m20_autonomy_spectrum.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
