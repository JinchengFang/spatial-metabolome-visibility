#!/usr/bin/env python3

import argparse, glob, os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score as ARI
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--datasets", default="IDHmut,GBM")
ap.add_argument("--res", type=float, default=50.0)
ap.add_argument("--sigma-um", type=float, default=100.0)
ap.add_argument("--max-feat", type=int, default=300)
ap.add_argument("--kmin", type=int, default=3)
ap.add_argument("--kmax", type=int, default=10)
ap.add_argument("--k-IDHmut", type=int, default=0)
ap.add_argument("--k-GBM", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--step", default="Step2", help="tables / figures go to 05_tables/<step> and 04_figures/<step>")
a = ap.parse_args(); P = C.paths(a.root)
TAB = os.path.join(os.path.dirname(P["TAB"]), a.step); FIG = os.path.join(os.path.dirname(P["FIG"]), a.step)
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7})
rng = np.random.default_rng(a.seed)


def select_features(var):
    v = var.copy()
    ok = v.get("use_downstream", pd.Series(True, index=v.index)).astype(bool)
    ok &= ~v.get("tier", pd.Series("C", index=v.index)).astype(str).isin(["isotope", "matrix/background", "inorganic/salt"])
    if "frac_pixels" in v:
        ok &= v.frac_pixels.fillna(0) >= 0.05
    v = v[ok]
    if "morans_i" in v:
        v = v.sort_values("morans_i", ascending=False)
    return v.index[: a.max_feat]


def label_of(var, idx):
    bn = var.loc[idx, "best_name"].fillna("").astype(str) if "best_name" in var else pd.Series("", index=idx)
    ba = var.loc[idx, "best_adduct"].fillna("").astype(str) if "best_adduct" in var else pd.Series("", index=idx)
    t = var.loc[idx, "tier"].astype(str) if "tier" in var else pd.Series("", index=idx)
    mz = var.loc[idx, "mz_cal"] if "mz_cal" in var else var.loc[idx, "mz"]
    return [f"{n} {d} [{tt}]" if n else f"m/z {m:.4f}" for n, d, tt, m in zip(bn, ba, t, mz)]


def grid_sample(X, x_um, y_um, feats_idx, x0=None, y0=None):
    """Pixels -> 50-µm grid means -> log1p -> smoothing -> z. Returns (bin coordinates, feature matrix, grid information)."""
    tic = np.asarray(X.sum(1)).ravel(); ok = tic > 0
    Xs = X[:, feats_idx]
    Xs = sp.diags(np.where(ok, np.median(tic[ok]) / np.maximum(tic, 1e-12), 0)) @ (Xs if sp.issparse(Xs) else sp.csr_matrix(Xs))
    x0 = x_um[ok].min() if x0 is None else x0; y0 = y_um[ok].min() if y0 is None else y0
    ix = np.floor((x_um - x0) / a.res).astype(int); iy = np.floor((y_um - y0) / a.res).astype(int)
    nx, ny = ix[ok].max() + 1, iy[ok].max() + 1
    lin = iy * nx + ix
    Bm = sp.csr_matrix((np.ones(ok.sum()), (lin[ok], np.where(ok)[0])), shape=(nx * ny, len(tic)))
    cnt = np.asarray(Bm.sum(1)).ravel()
    M = np.asarray((Bm @ Xs).todense()) / np.maximum(cnt, 1)[:, None]
    good = cnt >= max(1, (a.res / max(px_size, 1)) ** 2 / 3)
    F = np.full((ny, nx, M.shape[1]), np.nan)
    F.reshape(-1, M.shape[1])[good] = np.log1p(M[good])
    sig = a.sigma_um / a.res
    for j in range(F.shape[2]):
        F[:, :, j] = np.where(np.isfinite(F[:, :, j]), C.nconv(F[:, :, j], sig), np.nan)
    gy, gx = np.where(good.reshape(ny, nx))
    Z = F[gy, gx]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return gx, gy, Z, dict(x0=x0, y0=y0, nx=nx, ny=ny)


def fit_km(Zp, k, seed):
    """Full k-means (n_init = 10) below 200,000 points, MiniBatch k-means above."""
    from sklearn.cluster import KMeans
    if len(Zp) <= 200000:
        return KMeans(k, n_init=10, random_state=seed).fit(Zp)
    return MiniBatchKMeans(k, n_init=10, random_state=seed, batch_size=8192).fit(Zp)


def choose_k(Zp, samples, gx, gy, forced):
    rows = []
    multi = len(np.unique(samples)) > 1
    for k in range(a.kmin, a.kmax + 1):
        ref = fit_km(Zp, k, 0).predict(Zp)
        stab = []
        for b_ in range(10):
            sub = rng.choice(len(Zp), len(Zp) // 2, replace=False)
            stab.append(ARI(ref, fit_km(Zp[sub], k, b_ + 1).predict(Zp)))
        if multi:   # leave-one-patient-out transfer: fit on the other samples, predict the held-out one, ARI against the full labels (median)
            rep = float(np.median([ARI(ref[samples == s_], fit_km(Zp[samples != s_], k, 0).predict(Zp[samples == s_])) for s_ in np.unique(samples)]))
        else:
            fold = ((gx // 20) + (gy // 20)) % 2                    # 1-mm checkerboard (50-µm grid)
            m0 = fit_km(Zp[fold == 0], k, 0)
            rep = ARI(ref[fold == 1], m0.predict(Zp[fold == 1]))
        # spatial coherence: fraction of 4-neighbours with the same label within a sample
        coh = []
        for s in np.unique(samples):
            sel = samples == s; G = {}
            for x_, y_, l_ in zip(gx[sel], gy[sel], ref[sel]):
                G[(x_, y_)] = l_
            same = [G[(x_, y_)] == G.get((x_ + 1, y_), -1) for (x_, y_) in G if (x_ + 1, y_) in G]
            coh.append(np.mean(same) if same else np.nan)
        rows.append(dict(k=k, stability_ARI=float(np.mean(stab)), reproducibility=float(rep), spatial_coherence=float(np.nanmean(coh))))
        C.log(f"    k={k}: stability {np.mean(stab):.3f}, reproducibility {rep:.3f}, spatial coherence {np.nanmean(coh):.3f}")
    K = pd.DataFrame(rows)
    if forced:
        return forced, K
    okk = K[(K.stability_ARI >= 0.85) & (K.reproducibility >= (0.8 if multi else 0.85))]
    k = int(okk.k.max()) if len(okk) else int(K.sort_values("stability_ARI").k.iloc[-1])
    if k == a.kmax:
        C.log(f"    [note] selected k equals the upper limit {a.kmax}; consider a larger --kmax")
    return k, K


summary = {}
for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
    C.log(f"== {ds}")
    blocks = []
    if ds == "IDHmut":
        A = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"))
        cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"), backed="r")
        px_size = 10.0
        tis = ~A.obs["empty"].values.astype(bool)
        A = A[tis].copy()
        cx, cy = cel.obsm["spatial"][:, 0], cel.obsm["spatial"][:, 1]
        x0 = min(A.obs.x_um.min(), cx.min()); y0 = min(A.obs.y_um.min(), cy.min())   # identical to the Xenium grid of M3-02
        feats = select_features(A.var); fi = [A.var_names.get_loc(f) for f in feats]
        gx, gy, Z, gi = grid_sample(A.X.tocsr(), A.obs.x_um.values, A.obs.y_um.values, fi, x0, y0)
        blocks.append(("SBIAD1426", gx, gy, Z, gi)); var = A.var
    else:
        fs = sorted(glob.glob(os.path.join(P["PROC"], "ravi_paired", "*_maldi.h5ad")))
        use_vis = bool(fs)
        if not fs:
            fs = sorted(glob.glob(os.path.join(P["PROC"], "ravi_maldi_sections", "S*.h5ad")))
            C.log("  [note] no ravi_paired (M2-03 not run); niches are defined in section coordinates and cannot be compared with Visium")
        if not fs:
            C.log("  no GBM MALDI data, skipped"); continue
        px_size = 50.0
        A0 = ad.read_h5ad(fs[0], backed="r"); var = A0.var.copy(); A0.file.close()
        feats = select_features(var)
        for f in fs:
            A = ad.read_h5ad(f)
            sname = A.uns.get("patient", os.path.basename(f).replace("_maldi.h5ad", "").replace(".h5ad", ""))
            xu, yu = (A.obs.x_vis_um.values, A.obs.y_vis_um.values) if use_vis else (A.obs.x_local_um.values, A.obs.y_local_um.values)
            fi = [A.var_names.get_loc(x) for x in feats]
            X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
            gx, gy, Z, gi = grid_sample(X, xu, yu, fi)
            blocks.append((sname, gx, gy, Z, gi))
    samples = np.concatenate([[b_[0]] * len(b_[1]) for b_ in blocks])
    GX = np.concatenate([b_[1] for b_ in blocks]); GY = np.concatenate([b_[2] for b_ in blocks])
    Zall = np.vstack([b_[3] for b_ in blocks])
    C.log(f"  {len(blocks)} sections, {len(Zall):,} 50-µm bins, {Zall.shape[1]} features")
    pca = PCA(n_components=min(20, Zall.shape[1] - 1), random_state=0).fit(Zall)
    Zp = pca.transform(Zall)
    k, K = choose_k(Zp, samples, GX, GY, getattr(a, f"k_{ds}"))
    K.to_csv(os.path.join(TAB, f"metabolic_niche_k_selection_{ds}.tsv"), sep="\t", index=False)
    km = fit_km(Zp, k, 0); lab = km.predict(Zp)
    order = pd.Series(lab).value_counts().index.tolist()
    rename = {old: i + 1 for i, old in enumerate(order)}
    lab = np.array([rename[l_] for l_ in lab])
    C.log(f"  -> k = {k}; niche sizes: " + ", ".join(f"M{i}:{(lab == i).mean():.1%}" for i in range(1, k + 1)))
    # ---- LOSO
    if len(blocks) > 1:
        lrows = []
        for s in np.unique(samples):
            tr = samples != s
            m_ = fit_km(Zp[tr], k, 0)
            lrows.append(dict(held_out=s, ARI_vs_full=ARI(lab[~tr], m_.predict(Zp[~tr])), n=int((~tr).sum())))
        L = pd.DataFrame(lrows); L.to_csv(os.path.join(TAB, f"metabolic_niche_loso_{ds}.tsv"), sep="\t", index=False)
        C.log(f"  leave-one-section-out ARI: median {L.ARI_vs_full.median():.3f} (range {L.ARI_vs_full.min():.3f}-{L.ARI_vs_full.max():.3f})")
        summary[ds + "_loso_median_ARI"] = float(L.ARI_vs_full.median())
    # ---- signatures
    labs = label_of(var, feats)
    srows = []
    for n_ in range(1, k + 1):
        D = []
        for s in np.unique(samples):
            sel = samples == s; inn = sel & (lab == n_); out = sel & (lab != n_)
            if inn.sum() < 20 or out.sum() < 20:
                continue
            sd = np.sqrt((Zall[inn].var(0) + Zall[out].var(0)) / 2) + 1e-9
            D.append((Zall[inn].mean(0) - Zall[out].mean(0)) / sd)
        if not D:
            continue
        D = np.array(D); med = np.median(D, 0)
        cons = (np.sign(D) == np.sign(med)).sum(0) if len(D) > 1 else np.ones(D.shape[1])
        for j in np.argsort(-np.abs(med))[:15]:
            srows.append(dict(niche=f"M{n_}", feature=feats[j], label=labs[j], median_d=float(med[j]), n_samples=len(D),
                              n_consistent=int(cons[j]), consistent=bool(cons[j] >= np.ceil(len(D) / 2))))
    S = pd.DataFrame(srows); S.to_csv(os.path.join(TAB, f"metabolic_niche_signatures_{ds}.tsv"), sep="\t", index=False)
    # ---- save labels
    out = []
    for (sname, gx, gy, Z, gi) in blocks:
        sel = samples == sname
        out.append(pd.DataFrame(dict(sample=sname, ix=gx, iy=gy, x_um=gi["x0"] + (gx + 0.5) * a.res, y_um=gi["y0"] + (gy + 0.5) * a.res,
                                     niche=[f"M{l_}" for l_ in lab[sel]], PC1=Zp[sel, 0], PC2=Zp[sel, 1], PC3=Zp[sel, 2])))
    O = pd.concat(out, ignore_index=True); O.to_csv(os.path.join(P["PROC"], f"metabolic_niches_{ds}.tsv.gz"), sep="\t", index=False)
    np.save(os.path.join(P["PROC"], f"metabolic_niche_centroids_{ds}.npy"), km.cluster_centers_)
    summary[ds] = dict(k=k, n_bins=int(len(O)), features=int(len(feats)), k_table=K.to_dict(orient="records"))
    # ---- figures
    nb = len(blocks); nc = min(3, nb); nr = int(np.ceil(nb / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.5 * nc, 4 * nr), squeeze=False)
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    for axx, (sname, gx, gy, Z, gi) in zip(axs.ravel(), blocks):
        im = np.full((gi["ny"], gi["nx"]), np.nan); im[gy, gx] = lab[samples == sname]
        axx.imshow(im, cmap=cmap, vmin=0.5, vmax=cmap.N + 0.5, interpolation="nearest"); axx.set_title(sname); axx.axis("off")
    for axx in axs.ravel()[nb:]:
        axx.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(1, k + 1)]
    fig.legend(handles, [f"M{i}" for i in range(1, k + 1)], loc="lower center", ncol=k, fontsize=7)
    fig.suptitle(f"{ds}: metabolic niches from MALDI only (k={k}, σ={a.sigma_um:.0f} µm)"); fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, f"metabolic_niches_{ds}_maps.png"), dpi=140); plt.close(fig)
    if len(S):
        top = S[S.consistent].groupby("niche").head(6)
        Mx = top.pivot_table(index="label", columns="niche", values="median_d", aggfunc="first").fillna(0)
        fig, axx = plt.subplots(figsize=(1 + 0.6 * Mx.shape[1], 0.25 * Mx.shape[0] + 1.5))
        im = axx.imshow(Mx.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        axx.set_yticks(range(len(Mx))); axx.set_yticklabels(Mx.index, fontsize=6); axx.set_xticks(range(Mx.shape[1])); axx.set_xticklabels(Mx.columns)
        fig.colorbar(im, ax=axx, label="median Cohen's d"); axx.set_title(f"{ds}: niche metabolite signatures (consistent features)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"metabolic_niche_signatures_{ds}.png"), dpi=150); plt.close(fig)
C.save_json(summary, os.path.join(TAB, "metabolic_niches_summary.json"))
C.log(f"→ {TAB}/metabolic_niches_summary.json")
