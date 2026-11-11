#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200); ap.add_argument("--sigma-ctx", type=float, default=400.0)
ap.add_argument("--matrix", default="FMP10"); ap.add_argument("--area-override", default=""); ap.add_argument("--max-feat", type=int, default=200)
ap.add_argument("--n-rand", type=int, default=999); ap.add_argument("--step", default="Step12"); ap.add_argument("--min-sb", type=float, default=0.2)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
cfg = L.Config(P, min_px_half=1, min_bins=60); rng = np.random.default_rng(0)
DELTA = 268.1070
NEURO = {"dopamine": 153.0790, "3-methoxytyramine": 167.0946, "norepinephrine": 169.0739, "L-DOPA": 197.0688, "tyramine": 137.0841, "serotonin": 176.0950}
CAND = {**{f"{k} (FMP-10 ×1)": v + DELTA for k, v in NEURO.items()}, **{f"{k} (FMP-10 ×2, 2+)": (v + 2 * DELTA) / 2 for k, v in NEURO.items()}}
DA_KEYS = ("dopamine", "3-methoxytyramine", "norepinephrine", "L-DOPA", "tyramine")
SYN = ["TH", "DDC", "SLC6A3", "SLC18A2"]
OVR = dict(x.split(":", 1) for x in a.area_override.split(",") if ":" in x)
fasym = os.path.join(TAB, "m17_asymmetry_features.tsv")
ASYM = pd.read_csv(fasym, sep="\t") if os.path.exists(fasym) else None


def kernel(xy, s):
    tree = cKDTree(xy); pr = tree.sparse_distance_matrix(tree, 3 * s, output_type="coo_matrix")
    K = sp.csr_matrix((np.exp(-0.5 * (pr.data / s) ** 2), (pr.row, pr.col)), shape=(len(xy), len(xy))) + sp.eye(len(xy))
    return sp.diags(1 / np.asarray(K.sum(1)).ravel()) @ K


rows, frows = [], []
for S in L.load_all(cfg, ["SMA"]):
    sec = S.get("msi_section", S["sample"]); area = OVR.get(sec, S.get("brain_area", ""))
    if a.matrix.lower() not in str(S.get("matrix", "")).lower() and sec not in OVR:
        continue
    if area not in ("Striatum", "Substantia Nigra"):
        continue
    R = a.res; G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    Y0, cnt = L.pixel_means(S, G, None); YA, _ = L.pixel_means(S, G, S["px_half"] == 0); YB, _ = L.pixel_means(S, G, S["px_half"] == 1)
    ok = cnt >= 1; xy = G["xy"][ok]
    det = (np.expm1(Y0[ok]) > 0).mean(0); okf = np.where(det >= 0.05)[0]
    reps, _, _, _ = C.select_structured_features(C.zcols(Y0[ok][:, okf]), xy, a.max_feat, 0.95, seed=R)
    cols = list(okf[reps]); names = np.asarray(S["names"]).astype(str); mzs = np.asarray(S["mz"], float)
    # candidates (included regardless of structured-feature selection)
    M_ = None
    try:
        import anndata as ad
        fpd = os.path.join(P["PROC"], "sma", f"{sec}_paired.h5ad"); M_ = ad.read_h5ad(fpd, backed="r"); lo, hi = M_.var.win_lo.values, M_.var.win_hi.values
    except Exception:
        lo, hi = mzs - 0.002, mzs + 0.002
    cand = {}
    for nm, m in CAND.items():
        hit = np.where((lo - 0.003 <= m) & (hi + 0.003 >= m))[0]
        if len(hit):
            cand[int(hit[np.argmin(np.abs(mzs[hit] - m))])] = nm
    if ASYM is not None and area == "Striatum":
        f_ = ASYM[(ASYM["sample"] == S["sample"]) & ASYM.replicated & (ASYM.higher_in == "intact")]
        for fn in f_.feature.astype(str):
            j = np.where(names == fn)[0]
            if len(j):
                cand.setdefault(int(j[0]), "lesion-depleted (data-driven)")
    da = [j for j, nm in cand.items() if any(k in nm for k in DA_KEYS) or "lesion" in nm]
    cols = sorted(set(cols) | set(cand))
    Y = C.zcols(Y0[ok][:, cols]); _, SB = C.split_half_reliability(C.zcols(YA[ok][:, cols]), C.zcols(YB[ok][:, cols]))
    CB = G["CB"][np.where(ok)[0]]; Bk, LG, _ = L.tx_blocks(S, CB, G["ncell"][ok], None)
    X0 = np.column_stack([np.asarray(v, float) for v in Bk.values()]); X0 = X0[:, X0.std(0) > 1e-9]; K = kernel(xy, a.sigma_ctx)
    TC = C.zcols(np.column_stack([X0, K @ X0])); TC = TC[:, TC.std(0) > 1e-9]; folds = C.spatial_folds(xy, 1000, seed=R)
    r2 = C.r2_columns(Y, C.cv_ridge_gcv(TC, Y, folds)); ratio = np.where(SB >= a.min_sb, np.clip(r2, -1, None) / SB, np.nan)
    isda = np.array([c in da for c in cols])
    # synthesis-gene score vs random 4-gene sets (matched by mean-expression strata)
    gn = np.asarray(S["genes"]); gi = [int(np.where(gn == g)[0][0]) for g in SYN if g in set(gn)]
    if len(gi) >= 2 and isda.any():
        Zg = C.zcols(LG); mexp = LG.mean(0); score = Zg[:, gi].mean(1); dav = Y[:, isda].mean(1)
        rho = np.corrcoef(score, dav)[0, 1]
        bins_ = np.digitize(mexp, np.quantile(mexp[mexp > 0], np.linspace(0, 1, 11)[1:-1])); null = []
        for _ in range(a.n_rand):
            rs = [rng.choice(np.where(bins_ == bins_[g])[0]) for g in gi]; null.append(np.corrcoef(Zg[:, rs].mean(1), dav)[0, 1])
        p_syn = (np.sum(np.array(null) >= rho) + 1) / (len(null) + 1)
    else:
        rho, p_syn = np.nan, np.nan
    rr = dict(sample=S["sample"], section=sec, brain_area=area, mouse=S.get("mouse", ""), n_bins=int(ok.sum()), n_feat=len(cols), n_da=int(isda.sum()),
              da_features="; ".join(f"{mzs[c]:.4f}={cand[c]}" for c in cols if c in da)[:500], ratio_da=float(np.nanmedian(ratio[isda])) if isda.any() else np.nan,
              ratio_other=float(np.nanmedian(ratio[~isda])), R2_da=float(np.median(r2[isda])) if isda.any() else np.nan, R2_other=float(np.median(r2[~isda])),
              SB_da=float(np.nanmedian(SB[isda])) if isda.any() else np.nan, rho_synthesis_genes=rho, p_synthesis_genes=p_syn, n_syn_genes=len(gi))
    rr["relative_predictability"] = rr["ratio_da"] / rr["ratio_other"] if rr["ratio_other"] and np.isfinite(rr["ratio_other"]) else np.nan
    rows.append(rr)
    for k, c in enumerate(cols):
        frows.append(dict(sample=S["sample"], brain_area=area, feature=names[c], mz=mzs[c], candidate=cand.get(c, ""), dopaminergic=bool(isda[k]), SB=SB[k], R2_TC=r2[k]))
    C.log(f"  {S['sample']} ({area}, {S.get('mouse', '')}): {ok.sum()} bins; {isda.sum()} dopaminergic features; R²/SB dopaminergic {rr['ratio_da']:.2f} vs other {rr['ratio_other']:.2f} "
          f"(relative {rr['relative_predictability']:.2f}); r with TH/DDC/SLC6A3/SLC18A2 score = {rho:+.2f} (random gene sets p {p_syn:.3f})")
if not rows:
    C.log("no FMP-10 striatum / substantia nigra sections (run SMA_SET=neuro bash p2_step12_downloads.sh, then run_step12.sh 1-2); D54 undetermined"); sys.exit(0)
T = pd.DataFrame(rows); FR = pd.DataFrame(frows)
v = {}
for ar in ("Substantia Nigra", "Striatum"):
    d = T[T.brain_area == ar]
    v[ar] = dict(n=int(len(d)), rel_pred=float(d.relative_predictability.median()) if len(d) else np.nan,
                 frac_syn_sig=float((d.p_synthesis_genes < 0.05).mean()) if len(d) else np.nan, rho_syn=float(d.rho_synthesis_genes.median()) if len(d) else np.nan)
sn, st = v["Substantia Nigra"], v["Striatum"]
v["D54_soma_vs_terminal"] = bool(sn["n"] and st["n"] and sn["rel_pred"] >= 1.5 * st["rel_pred"] and sn["frac_syn_sig"] >= 0.5 and st["frac_syn_sig"] < 0.5) if sn["n"] and st["n"] else None
T.to_csv(os.path.join(TAB, "m17_soma_terminal.tsv"), sep="\t", index=False); FR.to_csv(os.path.join(TAB, "m17_soma_terminal_features.tsv"), sep="\t", index=False)
C.save_json(v, os.path.join(TAB, "m17_soma_terminal_verdict.json"))
C.log("soma vs terminal:\n" + T.drop(columns=["da_features"]).round(3).to_string(index=False) + "\ndecision: " + str(v))
fig, ax = plt.subplots(figsize=(3.6, 3.2))
for i, ar in enumerate(("Substantia Nigra", "Striatum")):
    d = T[T.brain_area == ar]; ax.scatter(np.full(len(d), i) + rng.uniform(-.08, .08, len(d)), d.relative_predictability, s=20, c=["#de2d26", "#3182bd"][i])
ax.axhline(1, c="k", ls="--", lw=.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["substantia nigra\n(soma: Th/Ddc mRNA)", "striatum\n(axon terminals)"])
ax.set_ylabel("dopaminergic vs other features\n(R² / ceiling, ratio)"); ax.set_title("same molecule, different compartment", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m17_soma_terminal.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
