#!/usr/bin/env python3

import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--n-perm", type=int, default=999); ap.add_argument("--n-boot", type=int, default=999)
ap.add_argument("--res", default="100,200,400"); ap.add_argument("--shift-step", default="Step13"); ap.add_argument("--step", default="Step20")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
RES = [int(x) for x in a.res.split(",")]; rng = np.random.default_rng(20)
SH = {}
fs = os.path.join(P["P2"], "05_tables", a.shift_step, "m11_registration_summary.tsv")
if os.path.exists(fs):
    rs = pd.read_csv(fs, sep="\t"); SH = {r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples() if r.cv_gain > 0.02}
MOL = {"dopamine": ["TH", "DDC", "SLC6A3", "SLC18A2"], "serotonin": ["TPH2", "SLC6A4", "FEV"]}
NAA_MZ = {"DHB": [176.0553, 198.0373, 214.0113], "9AA": [174.0408, 210.0175]}; NAA_SIDE = ["NAT8L"] + C.PROGRAMS["Neuron_cortex"].split()


def label_mirror(xy_s, lab_s, PC):
    """Mirror axis initialized from the annotation and refined with the joint score. Returns th, d, symmetry score, swap rate."""
    c = xy_s.mean(0); ci, cl = xy_s[lab_s == "intact"].mean(0), xy_s[lab_s == "lesioned"].mean(0); v = cl - ci
    th0 = np.arctan2(v[1], v[0]) + np.pi / 2; n0 = np.array([-np.sin(th0), np.cos(th0)]); d0 = float(((ci + cl) / 2 - c) @ n0)
    tree = cKDTree(xy_s); best = (-9, th0, d0, np.nan, np.nan)
    for th in th0 + np.deg2rad(np.arange(-20, 20.01, 2.0)):
        for d in d0 + np.arange(-600, 601, 50):
            q = C.reflect(xy_s, th, d, c); dd, j = tree.query(q); ok = dd < 75
            if ok.sum() < 0.3 * len(xy_s):
                continue
            sym = float(np.mean([np.corrcoef(PC[ok, k], PC[j[ok], k])[0, 1] for k in range(PC.shape[1])]))
            swap = float(np.mean(lab_s[ok] != lab_s[j[ok]]))
            s_ = sym + swap
            if s_ > best[0]:
                best = (s_, th, d, sym, swap)
    return best[1], best[2], best[3], best[4]


def explained_fraction(dY, dS, xy, R):
    """No-intercept, spatial-block cross-fitted dY ~ b dS. Returns mean(prediction)/mean(dY)."""
    if abs(dY.mean()) < 0.05 or dS.std() < 1e-9:
        return np.nan
    folds = C.spatial_folds(xy, 1000, seed=R); pred = np.zeros(len(dY))
    for f_ in np.unique(folds):
        tr, te = folds != f_, folds == f_; pred[te] = float(dS[tr] @ dY[tr] / max(dS[tr] @ dS[tr], 1e-9)) * dS[te]
    return float(pred.mean() / dY.mean())


def frac_inference(dY, dS, xy, ix, iy, R):
    obs = explained_fraction(dY, dS, xy, R)
    if not np.isfinite(obs):
        return obs, np.nan, (np.nan, np.nan)
    B = C._block_index(ix, iy, max(2, int(round(1000 / R)))); nul, bs = [], []
    for _ in range(a.n_perm):
        Bp = B[rng.permutation(B.shape[0])]; v_ = (B >= 0) & (Bp >= 0); iy_, is_ = B[v_], Bp[v_]
        nul.append(explained_fraction(dY[iy_], dS[is_], xy[iy_], R))
    for _ in range(a.n_boot):
        rows_ = B[rng.integers(0, B.shape[0], B.shape[0])]; ii = rows_[rows_ >= 0]
        if len(ii) > 20:
            bs.append(explained_fraction(dY[ii], dS[ii], xy[ii], R))
    nul, bs = np.array(nul, float), np.array(bs, float); nul, bs = nul[np.isfinite(nul)], bs[np.isfinite(bs)]
    p = float((np.sum(nul >= obs) + 1) / (len(nul) + 1)) if len(nul) else np.nan
    return obs, p, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) > 20 else (np.nan, np.nan)


import anndata as ad
cfg = L.Config(P, min_px_half=0, min_bins=40); SECS = [S for S in L.load_all(cfg, ["SMA"]) if (S.get("tx_labels") or {}).get("lesion") is not None]
AX = {}
for S in SECS:                                                                     # once per section: mirror axis (independent of bin size)
    lab_s = pd.Series(np.asarray(S["tx_labels"]["lesion"]).astype(str)).str.lower().map(lambda s: "intact" if s.startswith("intact") else ("lesioned" if s.startswith("lesion") else "NA")).values
    CTs = S["counts"]; tot_s = np.asarray(CTs.sum(1)).ravel(); LGs = (sp.diags(1e4 / np.maximum(tot_s, 1)) @ CTs).tocsr(); LGs.data = np.log1p(LGs.data)
    v_ = np.asarray(LGs.power(2).mean(0)).ravel() - np.asarray(LGs.mean(0)).ravel() ** 2; hv = np.argsort(-v_)[:2000]
    Zs = C.zcols(LGs[:, hv].toarray()); U, Sv, _ = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); PC = U[:, :10] * Sv[:10]
    th_old, d_old, s_old = C.mirror_axis(S["tx_xy"], PC); th, d, sym, swap = label_mirror(S["tx_xy"], lab_s, PC)
    AX[S["sample"]] = dict(th=th, d=d, sym=sym, swap=swap, th_old=th_old, d_old=d_old, sym_old=s_old, lab=lab_s)
    C.log(f"  {S['sample']}: annotation-constrained mirror axis {np.rad2deg(th) % 180:.1f} deg (symmetry {sym:.2f}, intact <-> lesioned swap rate {swap:.2f}); gene-PC axis {np.rad2deg(th_old) % 180:.1f} deg (symmetry {s_old:.2f})")
rows = []
for shift_on in (True, False):
    for R in RES:
        for S in SECS:
            mtx = str(S.get("matrix", "")).upper().replace("-", ""); kind = "FMP10" if "FMP" in mtx else ("DHB" if "DHB" in mtx else ("9AA" if "9AA" in mtx else None))
            if kind is None:
                continue
            G = L.make_grid(S, R, cfg)
            if G is None:
                continue
            Y0, cnt = L.pixel_means(S, G, None, SH.get(S["sample"], (0.0, 0.0)) if shift_on else (0.0, 0.0)); ok = cnt >= 1; xy = G["xy"][ok]; Y = Y0[ok]; det = (np.expm1(Y) > 0).mean(0)
            keys = np.asarray(G["keys"])[ok]; ix, iy = (keys % 100000).astype(int), (keys // 100000).astype(int); ix -= ix.min(); iy -= iy.min()
            kt = np.floor((S["tx_xy"] - G["o"]) / R).astype(np.int64); it = G["gi"].reindex(kt[:, 1] * 100000 + kt[:, 0]).fillna(-1).astype(int).values
            lab_s = AX[S["sample"]]["lab"]
            ni = np.bincount(it[(it >= 0) & (lab_s == "intact")], minlength=len(G["keys"]))[ok]; nl = np.bincount(it[(it >= 0) & (lab_s == "lesioned")], minlength=len(G["keys"]))[ok]
            blab = np.where(ni > nl, "intact", np.where(nl > ni, "lesioned", "NA"))
            dd, jj = cKDTree(xy).query(C.reflect(xy, AX[S["sample"]]["th"], AX[S["sample"]]["d"], S["tx_xy"].mean(0)))
            I = np.where((blab == "intact") & (dd < 0.75 * R) & (blab[jj] == "lesioned"))[0]; J = jj[I]
            if len(I) < 20:
                continue
            CB = G["CB"][np.where(ok)[0]].tocsc(); tot = np.asarray(CB.sum(1)).ravel(); gn = np.asarray(S["genes"]).astype(str)
            need = set(sum(MOL.values(), []) + NAA_SIDE); gi_ = [i for i, g in enumerate(gn) if g in need]
            LG = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CB[:, gi_]).toarray()); pos = {g: i for i, g in enumerate(gn[gi_])}; Zg = np.nan_to_num(C.zcols(LG))
            var = ad.read_h5ad(os.path.join(P["PROC"], "sma", f"{S['msi_section']}_paired.h5ad"), backed="r").var; mz, lo, hi = var.mz.values, var.win_lo.values, var.win_hi.values
            tol = lambda j, t: abs(mz[j] - t) / t * 1e6 <= max(5, (hi[j] - lo[j]) / 2 / t * 1e6 + 2)
            tests = []
            if kind == "FMP10":
                for mol, genes in MOL.items():
                    M, ns, _ = C.NEURO_PANEL[mol]; js = sorted({int(np.argmin(np.abs(mz - t))) for _, t in C.fmp10_ions(M, ns)})
                    js = [j for j in js if det[j] >= 0.05 and any(tol(j, t) for _, t in C.fmp10_ions(M, ns))]; tests.append((mol, js, genes))
            else:
                js = [int(np.argmin(np.abs(mz - t))) for t in NAA_MZ[kind]]; js = [j for j, t in zip(js, NAA_MZ[kind]) if tol(j, t) and det[j] >= 0.05]
                tests.append(("N-acetylaspartate", js, NAA_SIDE))
            for mol, js, genes in tests:
                gi = [pos[g] for g in genes if g in pos]
                if not js or not gi:
                    continue
                y = C.zcols(C.zcols(Y[:, js]).mean(1)[:, None])[:, 0]; syn = Zg[:, gi].mean(1); dY, dS = y[I] - y[J], syn[I] - syn[J]
                main = shift_on and R == 200
                if main:
                    fr, p, ci = frac_inference(dY, dS, xy[I], ix[I], iy[I], R)
                    rr, p2, _ = C.block_perm_corr(dY, dS, ix[I], iy[I], block=max(2, int(round(1000 / R))), n_perm=a.n_perm, seed=R) if dS.std() > 1e-9 else (np.array([[np.nan]]),) * 3
                    r_ = float(rr[0, 0])
                else:
                    fr, p, ci, r_ = explained_fraction(dY, dS, xy[I], R), np.nan, (np.nan, np.nan), np.nan
                rows.append(dict(shift=shift_on, res_um=R, sample=S["sample"], matrix=kind, brain_area=S.get("brain_area"), mouse=S.get("mouse", ""), molecule=mol, n_pairs=len(I),
                                 mean_dY=float(dY.mean()), mean_dSyn=float(dS.mean()), frac_explained=fr, p_frac=p, ci_lo=ci[0], ci_hi=ci[1], r_paired=r_))
        C.log(f"  variant shift={shift_on} {R} µm: done")
D = pd.DataFrame(rows); D.to_csv(os.path.join(TAB, "m22r_sections.tsv"), sep="\t", index=False)
M0 = D[(D["shift"]) & (D.res_um == 200)]
C.log("main analysis (shift applied, 200 µm):\n" + M0.round(4).to_string(index=False))
v = {}
da = M0[M0.molecule == "dopamine"]; sn, st = da[da.brain_area == "Substantia Nigra"], da[da.brain_area == "Striatum"]
if len(sn):
    ps = sn.p_frac.clip(1e-6, 1 - 1e-6); z = norm.isf(ps).sum() / np.sqrt(len(ps)); v["SN_stouffer_p"] = float(norm.sf(z))
    v["D63a_SN_fraction_positive"] = bool(v["SN_stouffer_p"] < 0.05 and (sn.frac_explained > 0.05).mean() >= 2 / 3)
    v["SN_animals"] = sn[["mouse", "n_pairs", "frac_explained", "p_frac", "ci_lo", "ci_hi", "r_paired"]].round(4).to_dict("records")
if len(sn) and len(st):
    mm = sn[["mouse", "frac_explained"]].merge(st[["mouse", "frac_explained"]], on="mouse", suffixes=("_SN", "_STR"))
    v["D63b_STR_below_SN_each_animal"] = bool(len(mm) and (mm.frac_explained_STR < mm.frac_explained_SN).all()); v["paired_animals"] = mm.round(4).to_dict("records")
    v["STR_animals"] = st[["mouse", "n_pairs", "frac_explained", "p_frac", "ci_lo", "ci_hi"]].round(4).to_dict("records")
sens = []
for (sh, R), g in D[D.molecule == "dopamine"].groupby(["shift", "res_um"]):
    s1, s2 = g[g.brain_area == "Substantia Nigra"].frac_explained.median(), g[g.brain_area == "Striatum"].frac_explained.median()
    sens.append(dict(shift=sh, res_um=R, n_SN=int((g.brain_area == "Substantia Nigra").sum()), n_STR=int((g.brain_area == "Striatum").sum()), SN_median=s1, STR_median=s2, contrast_holds=bool(s1 > s2 + 0.05)))
SE = pd.DataFrame(sens); SE.to_csv(os.path.join(TAB, "m22r_sensitivity.tsv"), sep="\t", index=False)
v["D63c_robust"] = bool(len(SE) and SE.contrast_holds.mean() >= 0.8); v["sensitivity"] = SE.round(3).to_dict("records")
for mol in ("serotonin", "N-acetylaspartate"):
    g = M0[M0.molecule == mol]
    if len(g):
        v[f"control_{mol}"] = g[["sample", "brain_area", "mean_dY", "frac_explained", "p_frac"]].round(3).to_dict("records")
v["mirror_axes"] = {k: dict(angle_label=round(float(np.rad2deg(x["th"]) % 180), 1), swap=round(x["swap"], 3), angle_gene_only=round(float(np.rad2deg(x["th_old"]) % 180), 1)) for k, x in AX.items()}
C.save_json(v, os.path.join(TAB, "m22r_verdict.json")); C.log("sensitivity:\n" + SE.round(3).to_string(index=False) + "\ndecision: " + json.dumps({k: v[k] for k in v if k.startswith("D63") or k == "SN_stouffer_p"}, ensure_ascii=False))
# figure
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
g = M0[M0.molecule.isin(["dopamine", "serotonin"])].copy(); g["lab"] = g.molecule.str[:3] + " " + g.brain_area.str.replace("Substantia Nigra", "SN").str.replace("Striatum", "STR") + " " + g.mouse
g = g.sort_values(["molecule", "brain_area", "mouse"]); x = np.arange(len(g))
cols = ["#de2d26" if (m == "dopamine" and b == "Substantia Nigra") else ("#3182bd" if m == "dopamine" else "#bdbdbd") for m, b in zip(g.molecule, g.brain_area)]
axs[0].bar(x, g.frac_explained, color=cols); axs[0].errorbar(x, g.frac_explained, yerr=[g.frac_explained - g.ci_lo, g.ci_hi - g.frac_explained], fmt="none", c="k", lw=.6)
for xi, p_ in zip(x, g.p_frac):
    if np.isfinite(p_):
        axs[0].text(xi, 0.01, f"P={p_:.3f}", rotation=90, fontsize=5, va="bottom", ha="center")
axs[0].axhline(0, c="k", lw=.5); axs[0].set_xticks(x); axs[0].set_xticklabels(g.lab, rotation=60, fontsize=5.5); axs[0].set_ylabel("lesion asymmetry explained by synthesis genes")
axs[0].set_title(f"Mirror-paired perturbation (200 µm, shift applied); SN Stouffer P = {v.get('SN_stouffer_p', np.nan):.3g}", fontsize=7)
if len(SE):
    lab_ = [f"{'shift' if r.shift else 'no shift'}\n{r.res_um} µm" for r in SE.itertuples()]; xx = np.arange(len(SE))
    axs[1].scatter(xx - .1, SE.SN_median, c="#de2d26", label="substantia nigra (soma)"); axs[1].scatter(xx + .1, SE.STR_median, c="#3182bd", label="striatum (terminals)")
    axs[1].set_xticks(xx); axs[1].set_xticklabels(lab_, fontsize=5.5); axs[1].axhline(0, c="k", lw=.5); axs[1].legend(fontsize=6, frameon=False)
    axs[1].set_title(f"Sensitivity: contrast holds in {100 * SE.contrast_holds.mean():.0f}% of variants", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m22r_explained_fraction.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
