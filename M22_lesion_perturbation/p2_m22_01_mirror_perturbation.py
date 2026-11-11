#!/usr/bin/env python3

import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200); ap.add_argument("--n-perm", type=int, default=1999)
ap.add_argument("--shift-step", default="Step13"); ap.add_argument("--step", default="Step19")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); R = a.res
cfg = L.Config(P, min_px_half=0, min_bins=40)
SH = {}
fs = os.path.join(P["P2"], "05_tables", a.shift_step, "m11_registration_summary.tsv")
if os.path.exists(fs):
    rs = pd.read_csv(fs, sep="\t"); SH = {r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples() if r.cv_gain > 0.02}
MOL = {"dopamine": ["TH", "DDC", "SLC6A3", "SLC18A2"], "serotonin": ["TPH2", "SLC6A4", "FEV"], "GABA": ["GAD1", "GAD2", "SLC32A1"]}
import anndata as ad
rows, pairs_plot = [], {}
for S in L.load_all(cfg, ["SMA"]):
    if "fmp" not in str(S.get("matrix", "")).lower() or S.get("brain_area") not in ("Striatum", "Substantia Nigra"):
        continue
    lab = (S.get("tx_labels") or {}).get("lesion")
    if lab is None:
        C.log(f"  {S['sample']}: no lesion annotation"); continue
    G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    Y0, cnt = L.pixel_means(S, G, None, SH.get(S["sample"], (0.0, 0.0))); ok = cnt >= 1; xy = G["xy"][ok]; Y = Y0[ok]; det = (np.expm1(Y) > 0).mean(0)
    keys = np.asarray(G["keys"])[ok]; ix, iy = (keys % 100000).astype(int), (keys // 100000).astype(int); ix -= ix.min(); iy -= iy.min()
    # lesion labels -> mirror pairs
    kt = np.floor((S["tx_xy"] - G["o"]) / R).astype(np.int64); it = G["gi"].reindex(kt[:, 1] * 100000 + kt[:, 0]).fillna(-1).astype(int).values
    ls = pd.Series(np.asarray(lab).astype(str)).str.lower()
    ni = np.bincount(it[(it >= 0) & ls.str.startswith("intact").values], minlength=len(G["keys"]))[ok]
    nl = np.bincount(it[(it >= 0) & ls.str.startswith("lesion").values], minlength=len(G["keys"]))[ok]
    blab = np.where(ni > nl, "intact", np.where(nl > ni, "lesioned", "NA"))
    CTs = S["counts"]; tot_s = np.asarray(CTs.sum(1)).ravel(); LGs = (sp.diags(1e4 / np.maximum(tot_s, 1)) @ CTs).tocsr(); LGs.data = np.log1p(LGs.data)
    v_ = np.asarray(LGs.power(2).mean(0)).ravel() - np.asarray(LGs.mean(0)).ravel() ** 2; hv = np.argsort(-v_)[:2000]
    Zs = C.zcols(LGs[:, hv].toarray()); U, Sv, _ = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    th, d, msc = C.mirror_axis(S["tx_xy"], U[:, :10] * Sv[:10])
    dd, jj = cKDTree(xy).query(C.reflect(xy, th, d, S["tx_xy"].mean(0)))
    I = np.where((blab == "intact") & (dd < 0.75 * R) & (blab[jj] == "lesioned"))[0]; J = jj[I]
    if len(I) < 30:
        C.log(f"  {S['sample']}: {len(I)} mirrored bin pairs < 30; skipped"); continue
    # genes (bins)
    CB = G["CB"][np.where(ok)[0]].tocsc(); tot = np.asarray(CB.sum(1)).ravel(); gn = np.asarray(S["genes"]).astype(str)
    need = set(sum(MOL.values(), []) + list(C.MSN_TARGET_GENES)); gi_ = [i for i, g in enumerate(gn) if g in need]
    LG = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CB[:, gi_]).toarray()); gk = gn[gi_]; pos = {g: i for i, g in enumerate(gk)}
    Zg = np.nan_to_num(C.zcols(LG))
    var = ad.read_h5ad(os.path.join(P["PROC"], "sma", f"{S['msi_section']}_paired.h5ad"), backed="r").var; mz, lo, hi = var.mz.values, var.win_lo.values, var.win_hi.values
    msn = [pos[g] for g in C.MSN_TARGET_GENES if g in pos]
    for mol, genes in MOL.items():
        M, ns, _ = C.NEURO_PANEL[mol]; js = []
        for _, t in C.fmp10_ions(M, ns):
            j = int(np.argmin(np.abs(mz - t)))
            if abs(mz[j] - t) / t * 1e6 <= max(5, (hi[j] - lo[j]) / 2 / t * 1e6 + 2) and det[j] >= 0.05:
                js.append(j)
        gi = [pos[g] for g in genes if g in pos]
        if not js or not gi:
            continue
        y = C.zcols(C.zcols(Y[:, sorted(set(js))]).mean(1)[:, None])[:, 0]; syn = Zg[:, gi].mean(1); tgt = Zg[:, msn].mean(1) if msn else np.zeros(len(y))
        dY, dS, dT = y[I] - y[J], syn[I] - syn[J], tgt[I] - tgt[J]
        out = dict(sample=S["sample"], brain_area=S["brain_area"], mouse=S.get("mouse", ""), molecule=mol, n_pairs=len(I), mirror_score=msc,
                   mean_dY=float(dY.mean()), d_dY=float(dY.mean() / (dY.std() + 1e-9)), mean_dSyn=float(dS.mean()), mean_dTarget=float(dT.mean()))
        if dS.std() > 1e-9:
            r, p2, _ = C.block_perm_corr(dY, dS, ix[I], iy[I], block=max(2, int(round(1000 / R))), n_perm=a.n_perm, seed=R)
            r = float(r[0, 0]); p1 = float(p2[0, 0] / 2 if r > 0 else 1 - p2[0, 0] / 2)
            folds = C.spatial_folds(xy[I], 1000, seed=R); pred = np.zeros(len(dY))
            for f_ in np.unique(folds):
                tr, te = folds != f_, folds == f_; b = float(dS[tr] @ dY[tr] / max(dS[tr] @ dS[tr], 1e-9)); pred[te] = b * dS[te]
            out.update(r_dY_dSyn=r, p_one_sided=p1, frac_asym_explained_by_syn=float(pred.mean() / dY.mean()) if abs(dY.mean()) > 0.05 else np.nan)
        else:
            out.update(r_dY_dSyn=np.nan, p_one_sided=np.nan, frac_asym_explained_by_syn=np.nan)
        if msn and dT.std() > 1e-9:
            r2_, p2_, _ = C.block_perm_corr(dY, dT, ix[I], iy[I], block=max(2, int(round(1000 / R))), n_perm=a.n_perm, seed=R + 1)
            out.update(r_dY_dTarget=float(r2_[0, 0]), p_target_two_sided=float(p2_[0, 0]))
        rows.append(out)
        if mol == "dopamine":
            pairs_plot[S["sample"]] = (dS, dY, S["brain_area"], S.get("mouse", ""))
    C.log(f"  {S['sample']} ({S['brain_area']}, {S.get('mouse', '')}): {len(I)} mirror pairs (mirror r {msc:.2f}); " + "; ".join(
        f"{r_['molecule']} dY {r_['mean_dY']:+.2f}, r(dY, dSyn) {r_['r_dY_dSyn']:+.2f} (p {r_['p_one_sided']:.3f}), explained by synthesis {r_['frac_asym_explained_by_syn']:+.2f}"
        for r_ in rows if r_["sample"] == S["sample"]))
T = pd.DataFrame(rows)
if not len(T):
    raise SystemExit("no FMP-10 sections with lesion annotation")
T.to_csv(os.path.join(TAB, "m22_pairs_summary.tsv"), sep="\t", index=False); C.log("paired perturbation test:\n" + T.round(3).to_string(index=False))
# within animal: delta synthesis in SN vs delta DA in STR
da = T[T.molecule == "dopamine"]
AL = da[da.brain_area == "Substantia Nigra"][["mouse", "mean_dSyn", "mean_dY"]].rename(columns={"mean_dSyn": "SN_dSyn", "mean_dY": "SN_dDA"}).merge(
    da[da.brain_area == "Striatum"][["mouse", "mean_dY", "mean_dSyn", "mean_dTarget"]].rename(columns={"mean_dY": "STR_dDA", "mean_dSyn": "STR_dSyn", "mean_dTarget": "STR_dTarget"}), on="mouse")
AL.to_csv(os.path.join(TAB, "m22_animal_link.tsv"), sep="\t", index=False); C.log("within animal (loss of nigral somata -> loss of striatal dopamine):\n" + AL.round(3).to_string(index=False))
v = {}
sn, st = da[da.brain_area == "Substantia Nigra"], da[da.brain_area == "Striatum"]
if len(sn):
    v["D62a_SN_pair_coupling"] = bool(((sn.r_dY_dSyn > 0) & (sn.p_one_sided < 0.05)).mean() >= 2 / 3)
if len(st):
    fs_sn, fs_st = sn.frac_asym_explained_by_syn.median(), st.frac_asym_explained_by_syn.median()
    v["D62b_STR_no_coupling"] = bool(((st.r_dY_dSyn > 0) & (st.p_one_sided < 0.05)).mean() <= 1 / 3 and (np.isnan(fs_st) or fs_st < fs_sn))
    v["frac_explained_median"] = dict(SN=float(fs_sn), STR=float(fs_st))
ht = T[T.molecule == "serotonin"]
if len(ht):
    v["D62c_5HT_control"] = bool(((ht.r_dY_dSyn > 0) & (ht.p_one_sided < 0.05)).mean() <= 1 / 3)
v["dopamine_detail"] = da[["sample", "brain_area", "mouse", "n_pairs", "mean_dY", "mean_dSyn", "r_dY_dSyn", "p_one_sided", "frac_asym_explained_by_syn"]].round(4).to_dict("records")
if len(AL) >= 3:
    v["animal_link_spearman"] = float(spearmanr(AL.SN_dSyn, AL.STR_dDA).correlation)
v["animal_link"] = AL.round(3).to_dict("records")
C.save_json(v, os.path.join(TAB, "m22_verdict.json")); C.log("decision: " + json.dumps(v, ensure_ascii=False, default=str)[:2500])
# figure
n_ = len(pairs_plot); fig, axs = plt.subplots(1, n_ + 1, figsize=(2.6 * (n_ + 1), 2.8))
for ax_, (smp, (dS, dY, ar, mo)) in zip(axs, sorted(pairs_plot.items(), key=lambda kv: (kv[1][2], kv[0]))):
    ax_.scatter(dS, dY, s=4, c="#de2d26" if ar == "Substantia Nigra" else "#3182bd", alpha=.6)
    rr = T[(T["sample"] == smp) & (T.molecule == "dopamine")].iloc[0]
    ax_.axhline(0, c="grey", lw=.4); ax_.axvline(0, c="grey", lw=.4); ax_.set_xlabel("Δ synthesis genes (intact − lesioned)"); ax_.set_ylabel("Δ dopamine")
    ax_.set_title(f"{'SN' if ar.startswith('Sub') else 'STR'} {mo}: r {rr.r_dY_dSyn:+.2f} (P {rr.p_one_sided:.3f})", fontsize=6)
ax_ = axs[-1]
if len(AL):
    ax_.scatter(AL.SN_dSyn, AL.STR_dDA, s=30, c="k")
    for r_ in AL.itertuples():
        ax_.text(r_.SN_dSyn, r_.STR_dDA, " " + str(r_.mouse), fontsize=6)
    ax_.set_xlabel("nigral Δ synthesis-gene score\n(soma loss, transcriptome)"); ax_.set_ylabel("striatal Δ dopamine\n(terminal loss, metabolome)"); ax_.set_title(f"same animal: soma → terminal (n = {len(AL)}, descriptive only)", fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m22_perturbation.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
