#!/usr/bin/env python3

import argparse, glob, gzip, json, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--data-dir", default=None); ap.add_argument("--res", default="200,400")
ap.add_argument("--min-sb", type=float, default=0.7); ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--step", default="Step22")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(22)
DD = a.data_dir or os.path.join(P["PUB"], "Li2026_COSIE_mouse_hippocampus_MALDI_Visium", "extracted")
RES = [int(x) for x in a.res.split(",")]
secs = sorted(d for d in glob.glob(os.path.join(DD, "**", "Section*"), recursive=True) if os.path.isdir(d) and "__MACOSX" not in d)
C.log(f"section folders: {secs}")


def read_visium(vd):
    mtx = glob.glob(os.path.join(vd, "filtered_feature_bc_matrix", "matrix.mtx*"))[0]
    M = sio.mmread(gzip.open(mtx) if mtx.endswith(".gz") else mtx).tocsc().T.tocsr()               # spot x gene
    bc = pd.read_csv(glob.glob(os.path.join(vd, "filtered_feature_bc_matrix", "barcodes.tsv*"))[0], header=None)[0].values
    ft = pd.read_csv(glob.glob(os.path.join(vd, "filtered_feature_bc_matrix", "features.tsv*"))[0], header=None, sep="\t")
    genes = ft[1].astype(str).str.upper().values
    tp = pd.read_csv(os.path.join(vd, "spatial", "tissue_positions.csv")); tp = tp.set_index("barcode").loc[bc]
    sf = json.load(open(os.path.join(vd, "spatial", "scalefactors_json.json")))
    return M, genes, tp[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float), 55.0 / sf["spot_diameter_fullres"]


def overlap(src, tgt, d):
    a_ = (cKDTree(tgt).query(src)[0] < d).mean(); b_ = (cKDTree(src).query(tgt)[0] < d).mean()
    return 2 * a_ * b_ / max(a_ + b_, 1e-9)


def register(src, tgt, d, n_cand=6):
    """Similarity-transform search: after shape scoring, n_cand candidates of different orientation (flip x 30-degree classes) are kept and refined.
    Returns [(f, score, parameters)] (coronal hippocampal sections are nearly symmetric; the caller chooses among candidates by cross-modal molecular consistency)."""
    cands = []
    sub = src[rng.choice(len(src), min(6000, len(src)), replace=False)]
    cs, ct = src.mean(0), tgt.mean(0); s0 = np.sqrt(np.trace(np.cov(tgt.T)) / np.trace(np.cov(src.T)))
    coarse = {}
    for flip in (False, True):
        for th in np.deg2rad(np.arange(0, 360, 5)):
            for s in s0 * np.array([0.9, 1.0, 1.1]):
                R_ = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ (np.diag([-1, 1]) if flip else np.eye(2))
                q = (sub - cs) @ R_.T * s + ct; sc = overlap(q, tgt, d); key = (flip, int(np.rad2deg(th) // 30))
                if sc > coarse.get(key, (-1,))[0]:
                    coarse[key] = (sc, (flip, th, s))
    top = sorted(coarse.values(), key=lambda t: -t[0])[:n_cand]
    for sc0, (flip, th0, s0b) in top:
        best = (sc0, (flip, th0, s0b, np.zeros(2)))
        for th in th0 + np.deg2rad(np.arange(-4, 4.01, 2)):
            for s in s0b * np.array([0.95, 1.0, 1.05]):
                for dx in np.arange(-2, 3) * d / 2:
                    for dy in np.arange(-2, 3) * d / 2:
                        R_ = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ (np.diag([-1, 1]) if flip else np.eye(2))
                        q = (sub - cs) @ R_.T * s + ct + np.array([dx, dy]); sc2 = overlap(q, tgt, d)
                        if sc2 > best[0]:
                            best = (sc2, (flip, th, s, np.array([dx, dy])))
        sc, (flip, th, s, tr) = best
        R_ = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ (np.diag([-1, 1]) if flip else np.eye(2))
        cands.append(((lambda X, R_=R_, s=s, tr=tr: (X - cs) @ R_.T * s + ct + tr), float(sc), dict(flip=flip, rot_deg=float(np.rad2deg(th) % 360), scale=float(s), shift=tr.tolist()),
                      dict(R=R_.tolist(), s=float(s), tr=tr.tolist(), cs=cs.tolist(), ct=ct.tolist())))
    return cands


def molecular_check(Xpc, xm_um, Mv, xyv, R=200.0):
    """Cross-modal consistency (for choosing registration candidates): mean spatial-block cross-validated R² of 30 gene PCs -> 10 MALDI PCs on 200-µm bins."""
    o = np.minimum(xm_um.min(0), xyv.min(0)); km = np.floor((xm_um - o) / R).astype(int); kv = np.floor((xyv - o) / R).astype(int)
    keym, keyv = km[:, 1] * 100000 + km[:, 0], kv[:, 1] * 100000 + kv[:, 0]; common = np.intersect1d(keym, keyv)
    if len(common) < 40:
        return -1.0
    gi = {k: i for i, k in enumerate(common)}; im = np.array([gi.get(k, -1) for k in keym]); iv = np.array([gi.get(k, -1) for k in keyv])
    Ym, _ = binmeans(Xpc[im >= 0], im[im >= 0], len(common)); vv = iv >= 0
    CB = (sp.csr_matrix((np.ones(vv.sum()), (iv[vv], np.where(vv)[0])), shape=(len(common), Mv.shape[0])) @ Mv).tocsr(); tot = np.asarray(CB.sum(1)).ravel(); ok = tot > 200
    LG = (sp.diags(1e4 / np.maximum(tot[ok], 1)) @ CB[ok]).tocsr(); LG.data = np.log1p(LG.data)
    mu = np.asarray(LG.mean(0)).ravel(); var = np.asarray(LG.multiply(LG).mean(0)).ravel() - mu ** 2; hv = np.argsort(-var)[:1000]
    U, S_, _ = np.linalg.svd(np.nan_to_num(C.zcols(LG[:, hv].toarray())), full_matrices=False)
    xyb = np.column_stack([common % 100000, common // 100000])[ok] * R; Y = C.zcols(Ym[ok])
    return float(np.mean(C.r2_columns(Y, C.cv_ridge_gcv(U[:, :30] * S_[:30], Y, C.spatial_folds(xyb, 1000, seed=0)))))


def binmeans(V, idx, nb):
    S = np.zeros((nb, V.shape[1]), np.float32); np.add.at(S, idx, V); n = np.bincount(idx, minlength=nb).astype(np.float32)
    return S / np.maximum(n, 1)[:, None], n


FR, SR, regs, REGJ = [], [], {}, {}
for sd in secs:
    name = os.path.basename(sd)
    cf, lf = glob.glob(os.path.join(sd, "MALDI", "cnts_*.tsv")), glob.glob(os.path.join(sd, "MALDI", "locs_*.tsv"))
    if not cf or not lf or not os.path.isdir(os.path.join(sd, "Visium")):
        continue
    Mv, genes, xyv_px, um_per_px = read_visium(os.path.join(sd, "Visium")); xyv = xyv_px * um_per_px
    Lc = pd.read_csv(lf[0], sep="\t"); Ct = pd.read_csv(cf[0], sep="\t", dtype=np.float32, engine="c")
    Ct = Ct.set_index(Ct.columns[0]); Lc = Lc.set_index(Lc.columns[0]).loc[Ct.index]
    X = Ct.values.astype(np.float32); xym = Lc[["x", "y"]].values.astype(float)
    tic = X.sum(1); okp = tic > 0; X, xym, tic = X[okp], xym[okp], tic[okp]
    X = np.log1p(X / tic[:, None] * np.median(tic))                                       # TIC normalization + log
    step = float(np.median(np.diff(np.unique(np.round(xym[:, 0], 3)))))                   # MALDI pixel spacing (original units)
    gx, gy = np.round((xym[:, 0] - xym[:, 0].min()) / step).astype(int), np.round((xym[:, 1] - xym[:, 1].min()) / step).astype(int); half = (gx + gy) % 2
    spacing_px = float(np.median(cKDTree(xyv_px).query(xyv_px, 2)[0][:, 1]))
    Uq, Sq, Vq = np.linalg.svd(X[rng.choice(len(X), min(8000, len(X)), replace=False)] - X.mean(0), full_matrices=False); Xpc = (X - X.mean(0)) @ Vq[:10].T
    cands = register(xym, xyv_px, 1.2 * spacing_px); scored = []
    for f_, sc_, prm_, T_ in cands:
        mc = molecular_check(Xpc, f_(xym) * um_per_px, Mv, xyv); scored.append((mc, sc_, prm_, f_, T_))
        C.log(f"     candidate flip={prm_['flip']} rotation {prm_['rot_deg']:.0f} deg scale {prm_['scale']:.3f}: shape {sc_:.3f}, cross-modal R² {mc:.3f}")
    scored = [t for t in scored if t[1] >= max(t_[1] for t_ in scored) - 0.05]               # among shape-acceptable candidates, choose by cross-modal consistency
    mc, sc, prm, f, T = max(scored, key=lambda t: t[0]); xm_um = f(xym) * um_per_px; regs[name] = (xym, xyv_px, f, sc); prm["molecular_R2"] = mc
    REGJ[name] = dict(**T, um_per_px=float(um_per_px), shape_score=float(sc), molecular_R2=float(mc), flip=bool(prm["flip"]), rot_deg=prm["rot_deg"], dir=sd)   # reused in Step 23
    C.log(f"  {name}: MALDI {X.shape[0]} pixels x {X.shape[1]} features (pixel spacing {step} original units); Visium {Mv.shape[0]} spots x {Mv.shape[1]} genes ({um_per_px:.3f} µm/pixel); "
          f"registration score {sc:.2f} {prm}; MALDI pixel about {step * prm['scale'] * um_per_px:.1f} µm")
    det = (X > 0).mean(0); keepf = det >= a.min_det; X = X[:, keepf]; fid = np.asarray(Ct.columns)[keepf]
    for R in RES:
        o = np.minimum(xm_um.min(0), xyv.min(0)); km = np.floor((xm_um - o) / R).astype(int); kv = np.floor((xyv - o) / R).astype(int)
        keym, keyv = km[:, 1] * 100000 + km[:, 0], kv[:, 1] * 100000 + kv[:, 0]
        common = np.intersect1d(keym, keyv); gi = {k: i for i, k in enumerate(common)}; nb = len(common)
        im, iv = np.array([gi.get(k, -1) for k in keym]), np.array([gi.get(k, -1) for k in keyv])
        mm_ = im >= 0
        Yf, nf = binmeans(X[mm_], im[mm_], nb); YA, nA = binmeans(X[mm_ & (half == 0)], im[mm_ & (half == 0)], nb); YB, nB = binmeans(X[mm_ & (half == 1)], im[mm_ & (half == 1)], nb)
        vv = iv >= 0; Bv = sp.csr_matrix((np.ones(vv.sum()), (iv[vv], np.where(vv)[0])), shape=(nb, Mv.shape[0])); CB = (Bv @ Mv).tocsr()
        ok = (nA >= 2) & (nB >= 2) & (np.asarray(CB.sum(1)).ravel() > 200)
        if ok.sum() < 60:
            C.log(f"     {R} µm: {ok.sum()} valid bins < 60; skipped"); continue
        CB = CB[ok]; tot = np.asarray(CB.sum(1)).ravel(); LG = (sp.diags(1e4 / tot) @ CB).tocsr(); LG.data = np.log1p(LG.data)
        mu = np.asarray(LG.mean(0)).ravel(); var = np.asarray(LG.multiply(LG).mean(0)).ravel() - mu ** 2; hv = np.argsort(-var)[:2000]
        Zg = C.zcols(LG[:, hv].toarray()); U, S_, _ = np.linalg.svd(np.nan_to_num(Zg), full_matrices=False); PCg = U[:, :30] * S_[:30]
        xyb = np.column_stack([common % 100000, common // 100000])[ok] * R
        Y, A_, B_ = C.zcols(Yf[ok]), C.zcols(YA[ok]), C.zcols(YB[ok])
        r_ab = np.array([np.corrcoef(A_[:, j], B_[:, j])[0, 1] for j in range(Y.shape[1])]); SB = 2 * r_ab / (1 + r_ab)
        folds = C.spatial_folds(xyb, 1000, seed=R); Yh = C.cv_ridge_gcv(C.zcols(PCg), Y, folds); r2tx = C.r2_columns(Y, Yh)
        EA, EB = A_ - Yh, B_ - Yh; rE = np.array([np.corrcoef(EA[:, j], EB[:, j])[0, 1] for j in range(Y.shape[1])])
        rel = np.where(np.nan_to_num(SB) >= a.min_sb)[0]; r2m = np.full(Y.shape[1], np.nan)
        for j in rel:                                                                        # cross-half internal MALDI reference
            oth = np.setdiff1d(np.arange(Y.shape[1]), [j]); Ua, Sa, _ = np.linalg.svd(np.nan_to_num(A_[:, oth]), full_matrices=False)
            r2m[j] = float(C.r2_columns(B_[:, [j]], C.cv_ridge_gcv(Ua[:, :20] * Sa[:20], B_[:, [j]], folds))[0])
        for j in range(Y.shape[1]):
            FR.append(dict(section=name, res_um=R, feature=fid[j], SB=SB[j], R2_tx=r2tx[j], ratio=r2tx[j] / SB[j] if SB[j] > 0 else np.nan, R2_maldi=r2m[j], r_resid_halves=rE[j]))
        g = pd.DataFrame(FR); g = g[(g.section == name) & (g.res_um == R) & (g.SB >= a.min_sb)]
        SR.append(dict(section=name, res_um=R, reg_score=sc, n_bins=int(ok.sum()), n_features=Y.shape[1], n_reliable=len(g), SB_median=float(np.nanmedian(SB)),
                       R2_tx_median=float(g.R2_tx.median()), ratio_median=float(g.ratio.median()), R2_maldi_median=float(g.R2_maldi.median()), r_resid_median=float(g.r_resid_halves.median())))
        C.log(f"     {R} µm: {ok.sum()} bins; {Y.shape[1]} features ({len(g)} reliable); median SB {np.nanmedian(SB):.2f}; reliable features R²_tx {g.R2_tx.median():.3f}, R²_tx/SB {g.ratio.median():.2f}, "
              f"R²_maldi {g.R2_maldi.median():.3f}, residual split-half r {g.r_resid_halves.median():.2f}")
C.save_json(REGJ, os.path.join(P["PROC"], "m23x_registration.json"))                      # read by M24 in Step 23 (same registration, no new search)
F, S = pd.DataFrame(FR), pd.DataFrame(SR)
F.to_csv(os.path.join(TAB, "m23x_features.tsv"), sep="\t", index=False); S.to_csv(os.path.join(TAB, "m23x_summary.tsv"), sep="\t", index=False)
v = dict(summary=S.round(3).to_dict("records"))
s2 = S[(S.res_um == 200) & (S.reg_score >= 0.6)] if len(S) else S
if len(s2):
    v["D65a_tx_partial"] = bool((s2.ratio_median < 0.75).all()); v["D65b_orthogonal_reproducible"] = bool((s2.r_resid_median > 0.3).all())
    v["D65c_maldi_internal_gt_tx"] = bool((s2.R2_maldi_median > s2.R2_tx_median).all())
else:
    v["D65a_tx_partial"] = v["D65b_orthogonal_reproducible"] = v["D65c_maldi_internal_gt_tx"] = "not testable (registration score < 0.6 or no valid section)"
v["reference_Vicari_same_section_ratio"] = 0.54
C.save_json(v, os.path.join(TAB, "m23x_verdict.json")); C.log("summary:\n" + S.round(3).to_string(index=False) + "\ndecision: " + json.dumps({k: v[k] for k in v if k.startswith("D65")}, ensure_ascii=False))
# Figure
if regs:
    fig, axs = plt.subplots(1, len(regs), figsize=(4 * len(regs), 3.6)); axs = np.atleast_1d(axs)
    for ax_, (nm, (xym, xyv_px, f, sc)) in zip(axs, regs.items()):
        q = f(xym); ax_.scatter(q[:, 0], q[:, 1], s=.2, c="#bdbdbd", label="MALDI pixels"); ax_.scatter(xyv_px[:, 0], xyv_px[:, 1], s=1.5, c="#de2d26", label="Visium spots")
        ax_.set_aspect("equal"); ax_.invert_yaxis(); ax_.set_title(f"{nm}: registration score {sc:.2f}", fontsize=7); ax_.legend(fontsize=5, markerscale=5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m23x_registration.png"), dpi=170); plt.close(fig)
if len(F):
    g = F[(F.res_um == 200) & (F.SB >= a.min_sb)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax_, col, lab in ((axs[0], "ratio", "R²_tx / SB (fraction of explainable variance)"), (axs[1], "R2_maldi", "R²_maldi (cross-half internal reference)"), (axs[2], "r_resid_halves", "split-half r of transcriptome-orthogonal residual")):
        for nm, gg in g.groupby("section"):
            ax_.hist(gg[col].clip(-0.5, 1.2), bins=30, alpha=.6, label=nm)
        ax_.set_xlabel(lab); ax_.legend(fontsize=6, frameon=False)
    axs[0].axvline(0.54, c="k", ls="--", lw=.7); axs[0].text(0.55, axs[0].get_ylim()[1] * .9, "Vicari same-section median", fontsize=5)
    fig.suptitle("External dataset (mouse hippocampus, another laboratory): reliable metabolite features at 200 µm", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m23x_framework.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
