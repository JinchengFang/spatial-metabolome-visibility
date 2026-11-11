#!/usr/bin/env python3

import argparse, json, os, re, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--n-rand", type=int, default=999); ap.add_argument("--n-null", type=int, default=999)
ap.add_argument("--n-perm", type=int, default=9999); ap.add_argument("--res", default="100,200,400"); ap.add_argument("--step", default="Step15")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
rng = np.random.default_rng(1); RES = [int(x) for x in a.res.split(",")]
NAA_RE = r"acetyl.?l?.?aspart(?!yl)"
NAA_MZ = {"DHB": [176.0553, 198.0373, 214.0113], "9AA": [174.0408, 210.0175]}
NEURON = C.PROGRAMS["Neuron_cortex"].split(); OLIGO = C.PROGRAMS["WM_oligodendrocyte"].split()
SIDE = {"neuronal": ["NAT8L"] + NEURON, "oligodendroglial": ["ASPA"] + OLIGO, "NAT8L": ["NAT8L"], "ASPA": ["ASPA"]}
MALIG = {"AC-like", "OPC-like", "MES-like", "NPC-like", "Progenitor", "Mixed-AC-like", "Malignant"}
NEUR = {"Excitatory", "Inhibitory", "Neuron"}; OLI = {"Oligodendrocyte"}
MALIG_PROG = ["AC_like", "OPC_like", "NPC_like", "MES1_inflammatory", "Hypoxia_MES2", "Cycling"]


def z(v):
    v = np.asarray(v, float); s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 0 else v * 0


def coef(y, X, k=-1):
    X1 = np.column_stack([np.ones(len(y)), X]); return float(np.linalg.lstsq(X1, y, rcond=None)[0][k])


def rotations(xy, n):
    ctr = xy.mean(0); tree = cKDTree(xy); out = []
    for th in rng.uniform(np.deg2rad(30), np.deg2rad(330), n):
        c_, s_ = np.cos(th), np.sin(th); out.append(tree.query((xy - ctr) @ np.array([[c_, -s_], [s_, c_]]).T + ctr)[1])
    return out


def bin_genes(S, G, ok, need):
    CB = G["CB"][np.where(ok)[0]].tocsc(); tot = np.asarray(CB.sum(1)).ravel()
    keep = np.where(np.asarray((CB > 0).mean(0)).ravel() >= 0.01)[0]; gn = np.asarray(S["genes"]).astype(str)
    keep = np.union1d(keep, [i for i, g in enumerate(gn) if g in need]).astype(int)
    LG = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CB[:, keep]).toarray()).astype(np.float32)
    return LG, gn[keep]


def gene_score_test(y, LG, gn, genes, xy, R):
    """r(y, gene-set score), p from expression-matched random gene sets, univariate spatial-block CV R²."""
    pos = {g: i for i, g in enumerate(gn)}; gi = [pos[g] for g in genes if g in pos]
    if not gi:
        return dict(n_genes=0, r=np.nan, p=np.nan, R2=np.nan)
    Zg = np.nan_to_num(C.zcols(LG)); sc = Zg[:, gi].mean(1)
    if sc.std() < 1e-9:
        return dict(n_genes=len(gi), r=np.nan, p=np.nan, R2=np.nan)
    r = float(np.corrcoef(sc, y)[0, 1]); mexp = LG.mean(0)
    qb = np.quantile(mexp[mexp > 0], np.linspace(0, 1, 11)[1:-1]); gb = np.digitize(mexp, qb)
    null = np.array([np.corrcoef(Zg[:, [rng.choice(np.where(gb == gb[g])[0]) for g in gi]].mean(1), y)[0, 1] for _ in range(a.n_rand)]); null = null[np.isfinite(null)]
    folds = C.spatial_folds(xy, 1000, seed=R); yy = y[:, None]
    r2 = float(C.r2_columns(yy, C.cv_ridge_gcv(C.zcols(sc[:, None]), yy, folds))[0])
    return dict(n_genes=len(gi), r=r, p=float((np.sum(null >= r) + 1) / (len(null) + 1)), R2=r2)


# ---------------- A mouse ----------------
rowsA = []
cfgS = L.Config(P, min_px_half=0, min_bins=40)
import anndata as ad
for S in L.load_all(cfgS, ["SMA"]):
    mtx = str(S.get("matrix", "")); key = "DHB" if "DHB" in mtx.upper() else ("9AA" if "9AA" in mtx.upper().replace("-", "") else None)
    if key is None:
        continue
    G = L.make_grid(S, 200, cfgS)
    if G is None:
        continue
    Y0, cnt = L.pixel_means(S, G, None, (0.0, 0.0)); ok = cnt >= 1
    var = ad.read_h5ad(os.path.join(P["PROC"], "sma", f"{S['msi_section']}_paired.h5ad"), backed="r").var
    mz, lo, hi = var.mz.values, var.win_lo.values, var.win_hi.values; idx, used = [], []
    for t in NAA_MZ[key]:
        j = int(np.argmin(np.abs(mz - t)))
        if abs(mz[j] - t) / t * 1e6 <= max(5, (hi[j] - lo[j]) / 2 / t * 1e6 + 2) and (np.expm1(Y0[ok, j]) > 0).mean() >= 0.05:
            idx.append(j); used.append(f"{mz[j]:.4f}")
    if not idx:
        C.log(f"  {S['sample']}: no NAA ion detected"); continue
    Yi = C.zcols(Y0[ok][:, idx]); y = C.zcols(Yi.mean(1)[:, None])[:, 0]; xy = G["xy"][ok]
    cons = float(np.nanmedian([np.corrcoef(Yi[:, i], Yi[:, k])[0, 1] for i in range(len(idx)) for k in range(i + 1, len(idx))])) if len(idx) > 1 else np.nan
    LG, gn = bin_genes(S, G, ok, set(sum(SIDE.values(), [])))
    row = dict(sample=S["sample"], matrix=key, brain_area=S.get("brain_area"), n_bins=len(y), naa_ions=",".join(used), ion_consistency=cons)
    for side, genes in SIDE.items():
        res = gene_score_test(y, LG, gn, genes, xy, 200)
        row.update({f"{side}_{k}": v for k, v in res.items()})
    pos = {g: i for i, g in enumerate(gn)}; Zg = np.nan_to_num(C.zcols(LG)); sc = Zg[:, [pos[g] for g in SIDE["neuronal"] if g in pos]].mean(1)
    det = (np.expm1(Y0[ok]) > 0).mean(0) >= 0.05; YA = C.zcols(Y0[ok][:, det]); rall = np.nan_to_num(YA.T @ C.zcols(sc[:, None])[:, 0] / len(sc))
    row["NAA_rank_pct_neuronal"] = float((rall < row["neuronal_r"]).mean()); row["n_features"] = int(det.sum())     # specificity: rank of NAA among all peaks (grey-matter molecules lean to the neuronal side)
    rowsA.append(row)
    C.log(f"  {S['sample']} ({key}): NAA ions {used} (consistency {cons:.2f}); r neuronal {row['neuronal_r']:+.2f} (p {row['neuronal_p']:.3f}), oligodendroglial {row['oligodendroglial_r']:+.2f} (p {row['oligodendroglial_p']:.3f}); "
          f"R² {row['neuronal_R2']:+.3f} vs {row['oligodendroglial_R2']:+.3f}; percentile of the neuronal correlation of NAA among {row['n_features']} peaks {row['NAA_rank_pct_neuronal']:.2f}")
A = pd.DataFrame(rowsA)
if len(A):
    A.to_csv(os.path.join(TAB, "m19c_mouse_naa.tsv"), sep="\t", index=False); C.log("mouse NAA:\n" + A.round(3).to_string(index=False))

# ---------------- B glioma, spatial interaction ----------------
rowsB, slopes = [], {}


def interaction(S, R, kind, base):
    G = L.make_grid(S, R, cfgG)
    if G is None:
        return
    Y0, cnt = L.pixel_means(S, G, None, (0.0, 0.0)); info = base(S, G)
    if info is None:
        return
    ok = (cnt >= 1) & info["keep"]
    if ok.sum() < 40:
        return
    names = np.asarray(S["names"]).astype(str); js = [j for j, n in enumerate(names) if re.search(NAA_RE, n, re.I)]
    xy = G["xy"][ok]; ne = z(info["ne"][ok]); mal = z(info["mal"][ok]); Bo = info["B"][ok]; NN = rotations(xy, a.n_null)
    for j in js:
        y = Y0[ok, j]
        if np.std(y) == 0 or (np.expm1(y) > 0).mean() < 0.2:
            continue
        y = z(y)
        b_ne = coef(y, np.column_stack([Bo, ne]))                                            # premise: NAA increases with neurons
        nb_ne = np.array([coef(y, np.column_stack([Bo, ne[nn]])) for nn in NN])
        X = lambda m: np.column_stack([Bo, ne, z(ne ** 2), m, ne * m])
        b_int = coef(y, X(mal)); nb_int = np.array([coef(y, X(mal[nn])) for nn in NN])
        rowsB.append(dict(kind=kind, sample=S["sample"], identity_high=S.get("_id_high", True), res_um=R, metabolite=names[j], n_bins=int(ok.sum()),
                          beta_neuron=b_ne, p_neuron_pos=float((np.sum(nb_ne >= b_ne) + 1) / (len(nb_ne) + 1)),
                          beta_interaction=b_int, p_int_neg=float((np.sum(nb_int <= b_int) + 1) / (len(nb_int) + 1)), null_sd=float(nb_int.std())))
        if R == 200 or (kind == "GBM_programs" and (kind, S["sample"]) not in slopes):                                   # slope of NAA ~ neurons within each malignant tertile
            q = pd.qcut(mal, 3, labels=False, duplicates="drop"); sl = []
            for t_ in sorted(set(q)):
                m_ = q == t_; Xs = np.column_stack([np.ones(m_.sum()), Bo[m_], ne[m_]])
                bb = np.linalg.lstsq(Xs, y[m_], rcond=None)[0]; e = y[m_] - Xs @ bb
                se = np.sqrt(e @ e / max(m_.sum() - Xs.shape[1], 1) * np.linalg.pinv(Xs.T @ Xs)[-1, -1]); sl.append((bb[-1], se))
            slopes[(kind, S["sample"])] = sl


def idh_base(S, G):
    if S.get("ctype") is None:
        return None
    ct = np.asarray(S["ctype"]).astype(str)
    kt = np.floor((S["tx_xy"] - G["o"]) / G["R"]).astype(np.int64); it = G["gi"].reindex(kt[:, 1] * 100000 + kt[:, 0]).fillna(-1).astype(int).values
    nb = len(G["keys"]); v = it >= 0
    cnt = lambda m: np.bincount(it[v & m], minlength=nb).astype(float)
    neur, oli, mal, tot = cnt(np.isin(ct, list(NEUR))), cnt(np.isin(ct, list(OLI))), cnt(np.isin(ct, list(MALIG))), cnt(np.ones(len(ct), bool))
    return dict(ne=np.log1p(neur), mal=mal / np.maximum(tot, 1), B=np.column_stack([z(np.log1p(oli)), z(np.log1p(tot))]), keep=(neur >= 1) & (tot >= 3))


def gbm_base(S, G):
    try:
        _, LG, _ = L.tx_blocks(S, G["CB"], G["ncell"], None)
    except Exception:
        return None
    sc, _ = C.score_programs(LG, pd.Index(S["genes"]), C.PROGRAMS, min_genes=3)
    if "Neuron_cortex" not in sc or not any(p in sc for p in MALIG_PROG):
        return None
    ol = z(sc["WM_oligodendrocyte"].values) if "WM_oligodendrocyte" in sc else np.zeros(len(sc))
    mal = np.column_stack([z(sc[p].values) for p in MALIG_PROG if p in sc]).mean(1)
    return dict(ne=sc["Neuron_cortex"].values, mal=mal, B=np.column_stack([ol, z(np.log1p(np.asarray(G["CB"].sum(1)).ravel()))]), keep=np.ones(len(sc), bool))


cfgG = L.Config(P, ion_set="strict", min_px_half=1, min_bins=60)
for S in L.load_all(cfgG, ["IDHmut"]):
    for R in RES:
        interaction(S, R, "IDHmut_cells", idh_base)
hi_ids = set()
try:
    hi_ids = {S["sample"] for S in L.load_all(L.Config(P, ion_set="strict", min_px_half=1, min_bins=60, include_uncertain=False), ["GBM"])}
except Exception:
    pass
cfgG = L.Config(P, ion_set="strict", min_px_half=1, min_bins=60, include_uncertain=True)
for S in L.load_all(cfgG, ["GBM"]):
    S["_id_high"] = S["sample"] in hi_ids; interaction(S, 200, "GBM_programs", gbm_base)
B_ = pd.DataFrame(rowsB)
if len(B_):
    B_.to_csv(os.path.join(TAB, "m19c_glioma_interaction.tsv"), sep="\t", index=False); C.log("glioma NAA x neurons x malignant:\n" + B_.round(3).to_string(index=False))

# ---------------- C CPTAC robustness ----------------
CS = []
dcp = os.path.join(P["PROC"], "cptac")
if os.path.exists(os.path.join(dcp, "metabolome_lcimputed.parquet")):
    met = pd.read_parquet(os.path.join(dcp, "metabolome_lcimputed.parquet")); prog = pd.read_csv(os.path.join(dcp, "programs_rna.tsv"), sep="\t", index_col=0)
    rna = pd.read_parquet(os.path.join(dcp, "rna.parquet")); com = met.index.intersection(prog.index).intersection(rna.index)
    met, prog, rna = met.loc[com], prog.loc[com], rna.loc[com]
    lg = lambda g: (np.log2(rna[g].values + 1) if rna[g].max() > 50 else rna[g].values).astype(float)
    naa = [c for c in met.columns if re.search(NAA_RE, str(c), re.I)]
    tg = {}
    if naa:
        tg["NAA"] = met[naa[0]].values.astype(float)
    if "NAT8L" in rna.columns:
        tg["NAT8L"] = lg("NAT8L")
    ne, ol = z(prog["Neuron_cortex"]), z(prog["WM_oligodendrocyte"]); mal = z(np.column_stack([z(prog[p]) for p in MALIG_PROG if p in prog]).mean(1))
    mg = [g for g in ("RBFOX3", "SNAP25", "SYT1", "STMN2", "GABRA1", "SLC17A7") if g in rna.columns]
    ne_g = z(np.nanmean(np.column_stack([z(lg(g)) for g in mg]), 1)) if mg else ne
    other = np.column_stack([z(prog[p]) for p in ("Reactive_astro", "Microglia", "Macrophage_TAM", "Vascular") if p in prog])
    rk = lambda v: z(rankdata(np.nan_to_num(v, nan=np.nanmedian(v))))
    VAR = {"base": (np.column_stack([ne, z(ne ** 2), ol]), mal), "neuron_linear": (np.column_stack([ne, ol]), mal),
           "neuron_cubic": (np.column_stack([ne, z(ne ** 2), z(ne ** 3), ol]), mal), "mature_neuron_genes": (np.column_stack([ne_g, z(ne_g ** 2), ol]), mal),
           "rank": (np.column_stack([rk(ne), z(rk(ne) ** 2), rk(ol)]), rk(mal)), "plus_astro_myeloid_vascular": (np.column_stack([ne, z(ne ** 2), ol, other]), mal)}
    for p in MALIG_PROG:
        if p in prog:
            VAR[f"program:{p}"] = (np.column_stack([ne, z(ne ** 2), ol]), z(prog[p]))
    for vn, (Bv, mv) in VAR.items():
        for tn, yv in tg.items():
            yv = rk(yv) if vn == "rank" else yv
            f = np.isfinite(yv) & np.isfinite(Bv).all(1) & np.isfinite(mv)
            if f.sum() < 30:
                continue
            y = z(yv[f]); Xb = np.column_stack([Bv[f], mv[f]]); b = coef(y, Xb)
            nb = np.array([coef(y, np.column_stack([Bv[f], rng.permutation(mv[f])])) for _ in range(a.n_perm)])
            CS.append(dict(variant=vn, target=tn, n=int(f.sum()), beta_malignant=b, p_neg=float((np.sum(nb <= b) + 1) / (a.n_perm + 1))))
    CS = pd.DataFrame(CS)
    if len(CS):
        CS.to_csv(os.path.join(TAB, "m19c_cptac_sensitivity.tsv"), sep="\t", index=False); C.log("CPTAC sensitivity:\n" + CS.round(4).to_string(index=False))
else:
    CS = pd.DataFrame()

# ---------------- decisions ----------------
v = {}
if len(A):
    v["D58a_NAA_neuronal"] = bool((A.neuronal_p < 0.05).mean() >= 2 / 3 and (A.neuronal_r > A.oligodendroglial_r).mean() >= 2 / 3)
    v["D58b_attribution_neuronal"] = bool((A.neuronal_R2 > A.oligodendroglial_R2).mean() >= 2 / 3)
    v["mouse_detail"] = A[["sample", "matrix", "naa_ions", "ion_consistency", "neuronal_r", "neuronal_p", "oligodendroglial_r", "oligodendroglial_p",
                           "NAT8L_r", "NAT8L_p", "ASPA_r", "ASPA_p", "neuronal_R2", "oligodendroglial_R2", "NAA_rank_pct_neuronal"]].round(4).to_dict("records")
if len(B_):
    i2 = B_[(B_.kind == "IDHmut_cells") & (B_.res_um == 200)]
    if len(i2):
        v["D58e_IDHmut_NAA_tracks_neurons"] = bool((i2.beta_neuron > 0).all() and (i2.p_neuron_pos < 0.05).all())
        v["D58c_IDHmut_interaction"] = bool((i2.beta_interaction < 0).all() and (i2.p_int_neg < 0.05).all())
    v["IDHmut_detail"] = B_[B_.kind == "IDHmut_cells"][["res_um", "beta_neuron", "p_neuron_pos", "beta_interaction", "p_int_neg"]].round(4).to_dict("records")
    g = B_[B_.kind == "GBM_programs"]
    if len(g):
        per = g.groupby("sample").beta_interaction.mean()
        v["D58d_GBM_interaction_neg"] = f"{int((per < 0).sum())}/{len(per)}"
        v["D58d_GBM_pass"] = bool((per < 0).sum() >= 4) if len(per) >= 6 else None
        v["GBM_detail"] = g[["sample", "identity_high", "metabolite", "beta_neuron", "p_neuron_pos", "beta_interaction", "p_int_neg"]].round(4).to_dict("records")
if len(CS):
    rob = CS[(CS.target == "NAA") & ~CS.variant.str.startswith("program:")]
    v["D57b_variants_pass"] = f"{int(((rob.beta_malignant < 0) & (rob.p_neg < 0.05)).sum())}/{len(rob)}"
    v["D57b_robust"] = bool(((rob.beta_malignant < 0) & (rob.p_neg < 0.05)).sum() >= 5) if len(rob) == 6 else None
    pr = CS[(CS.target == "NAA") & CS.variant.str.startswith("program:")]
    v["NAA_by_program"] = {r.variant.split(":")[1]: dict(beta=round(r.beta_malignant, 3), p_neg=round(r.p_neg, 4)) for r in pr.itertuples()}
v["tertile_slopes_NAA_on_neurons"] = {f"{k[0]} {k[1]}": [dict(slope=round(float(b), 3), se=round(float(e), 3)) for b, e in sl] for k, sl in slopes.items()}
C.save_json(v, os.path.join(TAB, "m19c_verdict.json")); C.log("decision: " + json.dumps(v, ensure_ascii=False, default=str)[:3000])

# ---------------- figure ----------------
fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
if len(A):
    x = np.arange(len(A))
    for k, (col, c_, lab) in enumerate((("neuronal_r", "#de2d26", "neuronal (NAT8L + neuron genes)"), ("oligodendroglial_r", "#3182bd", "oligodendroglial (ASPA + myelin genes)"))):
        axs[0].bar(x + (k - .5) * .38, A[col], .38, color=c_, label=lab)
        for i, p_ in enumerate(A[col.replace("_r", "_p")]):
            if p_ < 0.05:
                axs[0].text(x[i] + (k - .5) * .38, A[col].iloc[i], "*", ha="center", fontsize=9)
    axs[0].set_xticks(x); axs[0].set_xticklabels([s.split("-")[-1] for s in A["sample"]]); axs[0].axhline(0, c="k", lw=.5)
    axs[0].set_ylabel("r(NAA, gene score)"); axs[0].legend(fontsize=5, frameon=False); axs[0].set_title("Mouse same-section: NAA follows its synthesising side?", fontsize=7)
    axs[1].bar(x - .19, A.neuronal_R2, .38, color="#de2d26"); axs[1].bar(x + .19, A.oligodendroglial_R2, .38, color="#3182bd"); axs[1].axhline(0, c="k", lw=.5)
    axs[1].set_xticks(x); axs[1].set_xticklabels([s.split("-")[-1] for s in A["sample"]]); axs[1].set_ylabel("spatial-block CV R²"); axs[1].set_title("Attribution: synthesis side vs catabolism side", fontsize=7)
k_ = ("IDHmut_cells", next((S for (kk, S) in slopes if kk == "IDHmut_cells"), None))
if k_ in slopes:
    sl = np.array(slopes[k_]); axs[2].errorbar(np.arange(len(sl)), sl[:, 0], yerr=1.96 * sl[:, 1], fmt="o-", c="#de2d26")
    axs[2].set_xticks(np.arange(len(sl))); axs[2].set_xticklabels(["low", "mid", "high"][:len(sl)]); axs[2].axhline(0, c="grey", lw=.5)
    axs[2].set_xlabel("local malignant-cell fraction (tertile)"); axs[2].set_ylabel("slope of NAA on log neurons (± 95% CI)"); axs[2].set_title("IDH-mut (200 µm): NAA contributed per neuron", fontsize=7)
if len(CS):
    c = CS[CS.target == "NAA"].reset_index(drop=True); t = CS[CS.target == "NAT8L"].set_index("variant")
    axs[3].scatter(c.beta_malignant, np.arange(len(c)), c=["#de2d26" if p < 0.05 else "#fc9272" for p in c.p_neg], s=18, label="NAA")
    tt = t.reindex(c.variant); axs[3].scatter(tt.beta_malignant, np.arange(len(c)), c="#fd8d3c", marker="x", s=14, label="NAT8L mRNA")
    axs[3].set_yticks(np.arange(len(c))); axs[3].set_yticklabels(c.variant, fontsize=5); axs[3].axvline(0, c="k", lw=.5); axs[3].legend(fontsize=5, frameon=False)
    axs[3].set_xlabel("β malignant | neuron, oligo (CPTAC)"); axs[3].set_title("CPTAC: model variants and malignant programmes", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m19c_naa_compartment.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
