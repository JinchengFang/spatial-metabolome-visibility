#!/usr/bin/env python3

import argparse, os, sys
import numpy as np, pandas as pd, anndata as ad
from scipy.ndimage import affine_transform
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--datasets", default="IDHmut,GBM")
ap.add_argument("--res", type=float, default=50.0)
ap.add_argument("--windows", default="50,100,200,400")
ap.add_argument("--n-null", type=int, default=200)
ap.add_argument("--min-cells", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--include-uncertain", action="store_true", help="GBM: include patients without high-confidence identity (sensitivity analysis only)")
ap.add_argument("--step", default="Step2", help="tables / figures go to 05_tables/<step> and 04_figures/<step>")
a = ap.parse_args(); P = C.paths(a.root)
TAB = os.path.join(os.path.dirname(P["TAB"]), a.step); FIG = os.path.join(os.path.dirname(P["FIG"]), a.step)
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7})
WIN = [int(w) for w in a.windows.split(",")]
rng = np.random.default_rng(a.seed)


def mode_coarsen(L, f):
    """Mode of a label map (-1 = empty) in f x f blocks."""
    if f == 1:
        return L
    ny, nx = (L.shape[0] // f) * f, (L.shape[1] // f) * f
    Lc = L[:ny, :nx]; k = Lc.max() + 1
    if k <= 0:
        return np.full((ny // f, nx // f), -1)
    cnt = np.zeros((ny // f, nx // f, k))
    for c in range(k):
        cnt[:, :, c] = (Lc == c).reshape(ny // f, f, nx // f, f).sum((1, 3))
    out = cnt.argmax(2); out[cnt.sum(2) < f * f / 3] = -1
    return out


def rigid(L, theta, t, center):
    """Rotate a label map about center by theta and translate by t (bins), nearest neighbour."""
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    off = center - R @ center - R @ t
    return affine_transform(L.astype(float), R, offset=off, order=0, cval=-1, mode="constant").astype(int)


def conc(T, M):
    ok = (T >= 0) & (M >= 0)
    if ok.sum() < 50:
        return np.nan, np.nan, int(ok.sum())
    return NMI(T[ok], M[ok]), ARI(T[ok], M[ok]), int(ok.sum())


def analyse(sample, T, M, TF, MF, tnames, mnames, ds):
    """T, M: label grids (int, -1 = empty); TF / MF: (ny, nx, p) feature grids."""
    rows = []
    center = np.array(np.where(M >= 0)).mean(1)[::-1]   # (x, y)
    # null maps
    NULL = []
    tries = 0
    base_ov = ((T >= 0) & (M >= 0)).sum()
    while len(NULL) < a.n_null and tries < a.n_null * 5:
        tries += 1
        th = rng.uniform(0, 2 * np.pi); tt = rng.uniform(-1000, 1000, 2) / a.res
        Mn = rigid(M, th, tt, center)
        if ((T >= 0) & (Mn >= 0)).sum() >= 0.5 * base_ov:
            NULL.append(Mn)
    for w in WIN:
        f = max(1, int(round(w / a.res)))
        Tc, Mc = mode_coarsen(T, f), mode_coarsen(M, f)
        nmi, ari, n = conc(Tc, Mc)
        nn = np.array([conc(Tc, mode_coarsen(Mn, f))[0] for Mn in NULL])
        nn = nn[np.isfinite(nn)]
        z = (nmi - nn.mean()) / max(nn.std(), 0.01) if len(nn) >= 20 else np.nan     # z unreliable with too narrow / too small null (s.d. floor 0.01)
        p = (np.sum(nn >= nmi) + 1) / (len(nn) + 1) if len(nn) else np.nan
        rows.append(dict(dataset=ds, sample=sample, window_um=w, NMI=nmi, ARI=ari, n_cells=n, null_NMI_mean=nn.mean() if len(nn) else np.nan,
                         null_NMI_p95=np.percentile(nn, 95) if len(nn) else np.nan, z=z, p_perm=p))
        C.log(f"   {sample} @ {w} µm: NMI {nmi:.3f} (null {nn.mean():.3f}+-{nn.std():.3f}, z {z:.1f}, p {p:.3g}); ARI {ari:.3f}")
    # ---- TSMH
    ok = (T >= 0) & (M >= 0)
    Tv, Mv = T[ok], M[ok]; TFv, MFv = TF[ok], MF[ok]
    cT = np.array([TFv[Tv == t].mean(0) for t in np.unique(Tv)]); cM = np.array([MFv[Mv == m].mean(0) for m in np.unique(Mv)])
    dT = np.median([np.linalg.norm(x - y) for i, x in enumerate(cT) for y in cT[i + 1:]]) if len(cT) > 1 else 1
    dM = np.median([np.linalg.norm(x - y) for i, x in enumerate(cM) for y in cM[i + 1:]]) if len(cM) > 1 else 1
    trows = []
    for t in np.unique(Tv):
        st = Tv == t
        if st.sum() < 3 * a.min_cells:
            continue
        vc = pd.Series(Mv[st]).value_counts()
        q = vc[(vc >= a.min_cells) & (vc >= 0.1 * st.sum())].index.tolist()
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                m1, m2 = q[i], q[j]
                s1, s2 = st & (Mv == m1), st & (Mv == m2)
                Td = np.linalg.norm(TFv[s1].mean(0) - TFv[s2].mean(0)) / dT
                Md = np.linalg.norm(MFv[s1].mean(0) - MFv[s2].mean(0)) / dM
                nd, ndm = [], []
                for Mn in NULL:     # rigidly transformed metabolic maps induce random but spatially contiguous splits of t
                    Mnv = Mn[ok]; n1, n2 = st & (Mnv == m1), st & (Mnv == m2)
                    if n1.sum() >= a.min_cells and n2.sum() >= a.min_cells:
                        nd.append(np.linalg.norm(TFv[n1].mean(0) - TFv[n2].mean(0)) / dT)
                        ndm.append(np.linalg.norm(MFv[n1].mean(0) - MFv[n2].mean(0)) / dM)
                nd, ndm = np.array(nd), np.array(ndm)
                enough = len(nd) >= 20
                pT = (np.sum(nd >= Td) + 1) / (len(nd) + 1) if enough else np.nan
                pM = (np.sum(ndm >= Md) + 1) / (len(ndm) + 1) if enough else np.nan
                ratioT = Td / np.median(nd) if enough else np.nan
                # silent = metabolic difference above random splits (pM < 0.05) and transcriptional difference not above their median (positive evidence)
                if not enough:
                    cls = "insufficient null"
                elif pT < 0.05:
                    cls = "transcriptionally coupled"
                elif pM < 0.05 and Md >= 0.5 and ratioT <= 1.0:
                    cls = "transcriptionally silent"
                elif pM < 0.05:
                    cls = "undetermined"
                else:
                    cls = "metabolic difference not significant"
                trows.append(dict(dataset=ds, sample=sample, tniche=tnames[t], m1=mnames[m1], m2=mnames[m2], n1=int(s1.sum()), n2=int(s2.sum()),
                                  M_rel=Md, p_M=pM, T_rel=Td, T_rel_null_median=float(np.median(nd)) if enough else np.nan, T_ratio_to_null=ratioT,
                                  p_T=pT, n_null=len(nd), cls=cls))
    return rows, trows


def label_grid(codes, ix, iy, shape):
    L = np.full(shape, -1); L[iy, ix] = codes; return L


def feat_grid(F, ix, iy, shape):
    G = np.full(shape + (F.shape[1],), np.nan); G[iy, ix] = F; return G


ALL, TS, CTG = [], [], []
dsets = [d.strip() for d in a.datasets.split(",") if d.strip()]
for ds in dsets:
    fm = os.path.join(P["PROC"], f"metabolic_niches_{ds}.tsv.gz"); ft = os.path.join(P["PROC"], f"transcriptional_niches_{ds}.tsv.gz")
    if not (os.path.exists(fm) and os.path.exists(ft)):
        C.log(f"== {ds}: niche files missing (run M3-01 / M3-02 first), skipped"); continue
    Mdf = pd.read_csv(fm, sep="\t"); Tdf = pd.read_csv(ft, sep="\t")
    mcat = sorted(Mdf.niche.unique()); tcat = sorted(Tdf.tniche.unique())
    C.log(f"== {ds}: {len(mcat)} metabolic niches, {len(tcat)} transcriptional niches")
    samples = sorted(set(Mdf["sample"]) & set(Tdf["sample"]))
    if ds == "GBM":   # identity gate: only patients with high-confidence identity enter paired analyses (M2-05)
        IDT, allowed = C.ravi_identity(P, a.include_uncertain)
        drop = [s for s in samples if s not in allowed]
        src = IDT.source_table.iloc[0] if len(IDT) else "no identity table"
        C.log(f"   GBM identity gate ({src}{', including uncertain' if a.include_uncertain else ''}): {len(samples) - len(drop)} / {len(samples)} patients included"
              + (f"; excluded {drop}" if drop else ""))
        samples = [s for s in samples if s in allowed]
    for s in samples:
        md, td = Mdf[Mdf["sample"] == s], Tdf[Tdf["sample"] == s]
        tf_cols = [c for c in td.columns if c.startswith("prog_") or c.startswith("comp_")]
        tf_cols = [c for c in tf_cols if not c.startswith("prog_") or not any(c == "prog_" + k for k in C.METAB_PROGRAMS)]
        if ds == "IDHmut":
            ny, nx = int(max(md.iy.max(), td.iy.max()) + 1), int(max(md.ix.max(), td.ix.max()) + 1)
            M = label_grid(md.niche.map({c: i for i, c in enumerate(mcat)}).values, md.ix.values, md.iy.values, (ny, nx))
            T = label_grid(td.tniche.map({c: i for i, c in enumerate(tcat)}).values, td.ix.values, td.iy.values, (ny, nx))
            TFz = td[tf_cols].values; TFz = (TFz - np.nanmean(TFz, 0)) / (np.nanstd(TFz, 0) + 1e-9)
            TF = feat_grid(np.nan_to_num(TFz), td.ix.values, td.iy.values, (ny, nx))
            MF = feat_grid(md[["PC1", "PC2", "PC3"]].values, md.ix.values, md.iy.values, (ny, nx))
        else:
            x0 = min(md.x_um.min(), td.x_um.min()) - 100; y0 = min(md.y_um.min(), td.y_um.min()) - 100
            nx = int((max(md.x_um.max(), td.x_um.max()) - x0) / a.res) + 3; ny = int((max(md.y_um.max(), td.y_um.max()) - y0) / a.res) + 3
            mix = np.floor((md.x_um.values - x0) / a.res).astype(int); miy = np.floor((md.y_um.values - y0) / a.res).astype(int)
            M = label_grid(md.niche.map({c: i for i, c in enumerate(mcat)}).values, mix, miy, (ny, nx))
            MF = feat_grid(md[["PC1", "PC2", "PC3"]].values, mix, miy, (ny, nx))
            gy, gx = np.mgrid[0:ny, 0:nx]; cxy = np.column_stack([x0 + (gx.ravel() + .5) * a.res, y0 + (gy.ravel() + .5) * a.res])
            tree = cKDTree(td[["x_um", "y_um"]].values); dd, ii = tree.query(cxy)
            near = dd <= 60
            T = np.full(ny * nx, -1); T[near] = td.tniche.map({c: i for i, c in enumerate(tcat)}).values[ii[near]]; T = T.reshape(ny, nx)
            TFz = td[tf_cols].values; TFz = (TFz - TFz.mean(0)) / (TFz.std(0) + 1e-9)
            TF = np.full((ny * nx, len(tf_cols)), np.nan); TF[near] = TFz[ii[near]]; TF = TF.reshape(ny, nx, -1)
        rows, trows = analyse(s, T, M, TF, MF, tcat, mcat, ds)
        ALL += rows; TS += trows
        ok = (T >= 0) & (M >= 0)
        ct = pd.crosstab(pd.Series(np.array(tcat)[T[ok]], name="tniche"), pd.Series(np.array(mcat)[M[ok]], name="mniche"))
        ct["sample"] = s; ct["dataset"] = ds; CTG.append(ct.reset_index())
R = pd.DataFrame(ALL); R.to_csv(os.path.join(TAB, "concordance_by_window.tsv"), sep="\t", index=False)
TSM = pd.DataFrame(TS); TSM.to_csv(os.path.join(TAB, "tsmh_pairs.tsv"), sep="\t", index=False)
CT = pd.concat(CTG) if CTG else pd.DataFrame(); CT.to_csv(os.path.join(TAB, "concordance_contingency.tsv"), sep="\t", index=False)

summary = {}
for ds, d in R.groupby("dataset"):
    comb = d.groupby("window_um").apply(lambda g: pd.Series(dict(n=int(g.z.notna().sum()), stouffer_z=g.z.dropna().sum() / np.sqrt(max(1, g.z.notna().sum())),
                                                                p_combined=norm.sf(g.z.dropna().sum() / np.sqrt(max(1, g.z.notna().sum()))))))
    comb.to_csv(os.path.join(TAB, f"concordance_combined_{ds}.tsv"), sep="\t")
    C.log(f"[{ds}] combined across samples (Stouffer): " + "; ".join(f"{int(w_)} µm z {r.stouffer_z:.2f} p {r.p_combined:.3g}" for w_, r in comb.iterrows()))
    w = d.groupby("window_um").agg(NMI=("NMI", "median"), null=("null_NMI_mean", "median"), n_sig=("p_perm", lambda p: int((p < 0.05).sum())), n=("p_perm", "size"))
    summary[ds] = dict(by_window=w.reset_index().to_dict(orient="records"), combined=comb.reset_index().to_dict(orient="records"))
    C.log(f"[{ds}] concordance (median): " + "; ".join(f"{int(r.window_um)} µm NMI {r.NMI:.3f} vs null {r.null:.3f} ({r.n_sig}/{r.n} significant)" for r in w.reset_index().itertuples()))
if len(TSM):
    agg = TSM.groupby(["dataset", "tniche", "m1", "m2"]).agg(n_samples=("sample", "nunique"), n_silent=("cls", lambda c: int((c == "transcriptionally silent").sum())),
                                                          n_coupled=("cls", lambda c: int((c == "transcriptionally coupled").sum())), M_rel=("M_rel", "median"),
                                                          T_rel=("T_rel", "median"), p_T_median=("p_T", "median")).reset_index()
    agg.to_csv(os.path.join(TAB, "tsmh_summary.tsv"), sep="\t", index=False)
    for ds, d in agg.groupby("dataset"):
        summary.setdefault(ds, {})["tsmh"] = dict(n_pairs=int(len(d)), n_silent_any=int((d.n_silent > 0).sum()), n_coupled_any=int((d.n_coupled > 0).sum()),
                                                  recurrent_silent=d[(d.n_silent >= 2)][["tniche", "m1", "m2", "n_silent", "n_samples"]].to_dict(orient="records"))
        C.log(f"[{ds}] TSMH: {len(d)} (t, m1, m2) groups; {int((d.n_silent > 0).sum())} with a transcriptionally silent split, {int((d.n_coupled > 0).sum())} with a coupled split")

# ----------------------------------------------------------------------------- genes (GBM, spot level)
gene_rows = []
if "GBM" in dsets and len(TSM) and (TSM.dataset == "GBM").any():
    import scanpy as sc
    rec = TSM[(TSM.dataset == "GBM") & TSM.cls.isin(["transcriptionally silent", "transcriptionally coupled"])].groupby(["tniche", "m1", "m2"])["sample"].apply(list)
    rec = rec[rec.map(len) >= 2]
    Mdf = pd.read_csv(os.path.join(P["PROC"], "metabolic_niches_GBM.tsv.gz"), sep="\t")
    Tdf = pd.read_csv(os.path.join(P["PROC"], "transcriptional_niches_GBM.tsv.gz"), sep="\t")
    metab_genes = {g for v in C.METAB_PROGRAMS.values() for g in v.split()}
    for (t, m1, m2), pts in rec.items():
        zs = {}
        for pid in pts:
            f = os.path.join(P["PROC"], "ravi_paired", f"{pid}_visium.h5ad")
            if not os.path.exists(f):
                continue
            B = ad.read_h5ad(f)
            md = Mdf[Mdf["sample"] == pid]; tree = cKDTree(md[["x_um", "y_um"]].values)
            dd, ii = tree.query(B.obsm["spatial_um"]); ml = np.where(dd <= 50, md.niche.values[ii], "")
            tl = Tdf[Tdf["sample"] == pid].set_index("barcode").reindex(B.obs_names).tniche.values
            grp = np.where((tl == t) & (ml == m1), "g1", np.where((tl == t) & (ml == m2), "g2", ""))
            if (grp == "g1").sum() < 15 or (grp == "g2").sum() < 15:
                continue
            S = B[grp != ""].copy(); S.obs["grp"] = pd.Categorical(grp[grp != ""])
            sc.tl.rank_genes_groups(S, "grp", groups=["g1"], reference="g2", method="wilcoxon")
            r = sc.get.rank_genes_groups_df(S, "g1")
            zs[pid] = r.set_index("names")["scores"]
        if len(zs) < 2:
            continue
        Z = pd.DataFrame(zs).dropna()
        stouffer = Z.sum(1) / np.sqrt(Z.shape[1]); cons = (np.sign(Z).eq(np.sign(stouffer), axis=0)).sum(1)
        p = 2 * norm.sf(np.abs(stouffer))
        D = pd.DataFrame(dict(gene=Z.index, stouffer_z=stouffer.values, p=p, n_patients=Z.shape[1], n_consistent=cons.values))
        D["q"] = C.bh(D.p.values); D["metabolic_gene"] = D.gene.isin(metab_genes)
        D = D.sort_values("stouffer_z", key=np.abs, ascending=False).head(60)
        D.insert(0, "contrast", f"{t} | {m1} vs {m2}"); gene_rows.append(D)
if gene_rows:
    G = pd.concat(gene_rows); G.to_csv(os.path.join(TAB, "tsmh_genes_GBM.tsv"), sep="\t", index=False)
    C.log(f"[GBM] sub-region gene differences: {G.contrast.nunique()} contrasts, {int((G.q < 0.05).sum())} genes with q < 0.05 ({int(((G.q < 0.05) & G.metabolic_gene).sum())} metabolic genes)")
C.save_json(summary, os.path.join(TAB, "concordance_summary.json"))

# ----------------------------------------------------------------------------- figure
if len(R):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ds, d in R.groupby("dataset"):
        for s, dd in d.groupby("sample"):
            axs[0].plot(dd.window_um, dd.NMI, "o-", alpha=.6, label=f"{ds}:{s}")
            axs[0].fill_between(dd.window_um, dd.null_NMI_mean, dd.null_NMI_p95, alpha=.1)
    axs[0].set_xscale("log"); axs[0].set_xlabel("window (µm)"); axs[0].set_ylabel("NMI (metabolic vs transcriptional niches)")
    axs[0].legend(fontsize=5); axs[0].set_title("concordance vs rigid-transform null (shaded)")
    if len(CT):
        for axx, ds in zip(axs[1:], sorted(CT.dataset.unique())[:2]):
            c = CT[CT.dataset == ds].drop(columns=["sample", "dataset"]).groupby("tniche").sum()
            cn = c.div(c.sum(1), axis=0)
            im = axx.imshow(cn.values, cmap="viridis", aspect="auto", vmin=0, vmax=1); fig.colorbar(im, ax=axx, label="fraction of T-niche")
            axx.set_yticks(range(len(cn))); axx.set_yticklabels(cn.index, fontsize=5); axx.set_xticks(range(cn.shape[1])); axx.set_xticklabels(cn.columns)
            axx.set_title(f"{ds}: transcriptional (rows) × metabolic (cols)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "concordance_overview.png"), dpi=150); plt.close(fig)
if len(TSM):
    fig, axx = plt.subplots(figsize=(5, 4))
    lab_en = {"metabolic difference not significant": "metabolic diff. n.s."}
    for cls, col in (("transcriptionally silent", "#2c7fb8"), ("transcriptionally coupled", "#c0392b"), ("undetermined", "#e69f00"), ("metabolic difference not significant", "#999")):
        d = TSM[TSM.cls == cls]
        axx.scatter(d.T_ratio_to_null, d.M_rel, s=18, c=col, alpha=.7, label=lab_en.get(cls, cls))
    axx.axvline(1, ls=":", c="grey")
    axx.axhline(0.5, ls="--", c="grey"); axx.set_xlabel("transcriptional difference / random-split median"); axx.set_ylabel("metabolic difference (relative)")
    axx.legend(fontsize=6); axx.set_title("sub-regions of the same transcriptional niche split by metabolism")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tsmh_scatter.png"), dpi=150); plt.close(fig)
