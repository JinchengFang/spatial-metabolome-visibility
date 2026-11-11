#!/usr/bin/env python3

import argparse, glob, os, re, sys, zipfile
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None)
a = ap.parse_args(); P = C.paths(a.root)
plt.rcParams.update({"font.size": 7})
PAT = re.compile(r"UKF[\s_#-]*0*(\d{2,4})", re.I)
pid = lambda s: (lambda m: f"UKF{int(m.group(1))}" if m else None)(PAT.search(str(s)))
SHOW = ["glycerophosphocholine", "phosphocholine", "choline", "lactate", "2-hydroxyglutarate", "N-acetylaspartate", "creatine",
        "glutamate", "taurine", "heme b", "ST 42:2;O2 (sulfatide 24:1)", "PC 34:1", "PI 38:4", "CE 18:1", "FA 22:6", "hexose"]

# ----------------------------------------------------------------------------- 1. MALDI QC plots and sample rows
rows = []
for f in sorted(glob.glob(os.path.join(P["PROC"], "ravi_maldi", "*.h5ad"))):
    A = ad.read_h5ad(f); name = os.path.basename(f)[:-5]
    src = A.uns.get("source", {}); cal = A.uns.get("mass_calibration", {})
    tg_f = os.path.join(P["TAB"], f"{name}_targets.tsv")
    tg = pd.read_csv(tg_f, sep="\t") if os.path.exists(tg_f) else pd.DataFrame(columns=["metabolite", "verdict", "peak"])
    X = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
    tissue = A.obs["tissue"].values.astype(bool); tic = A.obs["tic"].values
    x = A.obs.x_px.values - A.obs.x_px.min(); y = A.obs.y_px.values - A.obs.y_px.min()
    def img(v, m=None):
        G = np.full((y.max() + 1, x.max() + 1), np.nan); sel = tissue if m is None else m; G[y[sel], x[sel]] = v[sel]; return G
    Xn = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1e-9) * np.median(X[tissue].sum(1)))
    use = A.var["use_downstream"].values.astype(bool) if "use_downstream" in A.var else np.ones(A.n_vars, bool)
    from sklearn.decomposition import PCA
    Z = Xn[tissue][:, use]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    pcs = PCA(3, random_state=0).fit_transform(Z) if Z.shape[1] >= 3 else np.zeros((Z.shape[0], 3))
    rgb = np.zeros((y.max() + 1, x.max() + 1, 3))
    for k in range(3):
        v = pcs[:, k]; lo, hi = np.percentile(v, [2, 98]); rgb[y[tissue], x[tissue], k] = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    panels = [("log10 TIC", img(np.log10(tic + 1), np.ones_like(tissue))), ("tissue mask", img(tissue.astype(float), np.ones_like(tissue))), ("PC1-3 RGB", rgb)]
    for m in SHOW:
        r = tg[(tg.metabolite == m) & tg.get("peak", pd.Series(dtype=str)).notna()] if "peak" in tg else pd.DataFrame()
        if len(r) and r.peak.iloc[0] in A.var_names:
            j = list(A.var_names).index(r.peak.iloc[0])
            panels.append((f"{m}\n{r.adduct.iloc[0]} {r.mz.iloc[0]} [{r.tier.iloc[0]}]", img(Xn[:, j])))
    nc = 6; nr = int(np.ceil(len(panels) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.6 * nr * (y.max() + 1) / max(x.max() + 1, 1) + 0.6), squeeze=False)
    for axx, (t, im) in zip(axs.ravel(), panels):
        if im.ndim == 3:
            axx.imshow(im)
        else:
            lo, hi = np.nanpercentile(im, [1, 99]) if np.isfinite(im).any() else (0, 1); axx.imshow(im, cmap="viridis", vmin=lo, vmax=hi)
        axx.set_title(t, fontsize=6); axx.axis("off")
    for axx in axs.ravel()[len(panels):]:
        axx.axis("off")
    fig.suptitle(f"Ravi 2022 MALDI · {name} · {src.get('polarity', '')} · {A.uns.get('pixel_size_um', '')} µm"); fig.tight_layout()
    fig.savefig(os.path.join(P["FIG"], f"ravi_{name}_maps.png"), dpi=150); plt.close(fig)
    tiers = A.var["tier"].value_counts().to_dict() if "tier" in A.var else {}
    v = lambda m: (tg.loc[tg.metabolite == m, "verdict"].iloc[0] if (tg.metabolite == m).any() else "")
    rows.append(dict(dataset="Ravi2022_GBM", sample=name, patient=pid(name), maldi_file=src.get("file", ""), polarity=src.get("polarity", ""),
                     instrument=src.get("instrument", ""), matrix=src.get("matrix", ""), pixel_size_um=A.uns.get("pixel_size_um", ""),
                     mz_range=f"{src.get('mz_min', np.nan):.1f}-{src.get('mz_max', np.nan):.1f}", pixels=A.n_obs, tissue_pixels=int(tissue.sum()),
                     n_peaks=A.n_vars, tier_A=tiers.get("A", 0), tier_B=tiers.get("B", 0), tier_C=tiers.get("C", 0),
                     mass_error_median_ppm=cal.get("median_ppm"), tol_ppm=cal.get("tolerance_ppm"),
                     **{f"detect_{k}": v(m) for k, m in (("2HG", "2-hydroxyglutarate"), ("lactate", "lactate"), ("GPC", "glycerophosphocholine"),
                                                         ("NAA", "N-acetylaspartate"), ("heme", "heme b"), ("sulfatide", "ST 42:2;O2 (sulfatide 24:1)"))}))
    C.log(f"{name}: {A.n_obs:,} pixels (tissue {tissue.sum():,}), {A.n_vars} peaks, tier A/B = {tiers.get('A', 0)}/{tiers.get('B', 0)}")
maldi = pd.DataFrame(rows)

# ----------------------------------------------------------------------------- 2. Visium / scRNA samples
vis_rows = []
zp = os.path.join(P["RAVI"], "10XVisium_2.zip")
if os.path.exists(zp):
    try:
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        samples = {}
        for n in names:
            parts = [p_ for p_ in n.split("/") if p_]
            hit = next((p_ for p_ in parts if pid(p_)), None)
            if hit:
                d = samples.setdefault(hit, dict(n_files=0, has_h5=False, has_hires=False, has_he=False))
                d["n_files"] += 1
                d["has_h5"] |= n.endswith("filtered_feature_bc_matrix.h5")
                d["has_hires"] |= n.endswith("tissue_hires_image.png")
                d["has_he"] |= bool(re.search(r"(H&E|HE)/.+\.(tif|tiff|png|jpg|svs|ndpi)$", n, re.I))
        for s, d in samples.items():
            vis_rows.append(dict(source="10XVisium_2.zip", visium_sample=s, patient=pid(s), **d))
        C.log(f"10XVisium_2.zip: {len(samples)} sample folders")
    except zipfile.BadZipFile as e:
        C.log(f"[warning] cannot read 10XVisium_2.zip: {e}")


def obs_samples(h5, tag):
    if not os.path.exists(h5):
        return []
    try:
        B = ad.read_h5ad(h5, backed="r"); obs = B.obs.copy(); shape = B.shape; B.file.close()
    except Exception as e:
        C.log(f"[warning] cannot read {h5}: {e}"); return []
    cols = [c for c in obs.columns if obs[c].nunique() <= 500]
    best = max(cols, key=lambda c: obs[c].astype(str).map(lambda v: pid(v) is not None).mean() + (0.1 if re.search("sample|batch|patient|library", c, re.I) else 0), default=None)
    C.log(f"{tag}: {shape[0]:,} x {shape[1]:,}; obs columns: {', '.join(obs.columns[:25])}; sample column -> {best}")
    if best is None:
        return []
    vc = obs[best].astype(str).value_counts()
    return [dict(source=os.path.basename(h5), column=best, h5ad_sample=k, patient=pid(k), n=int(v)) for k, v in vc.items()]


zdir = os.path.join(P["RAVI"], "zenodo16505469_h5ad")
st = pd.DataFrame(obs_samples(os.path.join(zdir, "spatial-raw-data.h5ad"), "spatial_h5ad"))
sc = pd.DataFrame(obs_samples(os.path.join(zdir, "single-cell-tumor-data.h5ad"), "scRNA_h5ad"))
vis = pd.DataFrame(vis_rows)
for t, nm in ((vis, "ravi_visium_zip_samples.tsv"), (st, "ravi_spatial_h5ad_samples.tsv"), (sc, "ravi_scRNA_h5ad_samples.tsv")):
    if len(t):
        t.to_csv(os.path.join(P["TAB"], nm), sep="\t", index=False)

# ----------------------------------------------------------------------------- 3. matching + sample table
pts = sorted(set(maldi.get("patient", pd.Series(dtype=str)).dropna()) | set(vis.get("patient", pd.Series(dtype=str)).dropna())
             | set(st.get("patient", pd.Series(dtype=str)).dropna()), key=lambda s: int(re.sub(r"\D", "", s) or 0))
match = []
for p_ in pts:
    match.append(dict(patient=p_,
                      maldi_samples=";".join(maldi.loc[maldi.patient == p_, "sample"]) if len(maldi) else "",
                      visium_zip_samples=";".join(vis.loc[vis.patient == p_, "visium_sample"]) if len(vis) else "",
                      visium_h5ad_samples=";".join(f"{r.h5ad_sample}({r.n})" for r in st[st.patient == p_].itertuples()) if len(st) else "",
                      scRNA_h5ad_samples=";".join(f"{r.h5ad_sample}({r.n})" for r in sc[sc.patient == p_].itertuples()) if len(sc) else ""))
match = pd.DataFrame(match)
if len(match):
    match["MALDI_and_Visium"] = (match.maldi_samples != "") & ((match.visium_zip_samples != "") | (match.visium_h5ad_samples != ""))
    match.to_csv(os.path.join(P["TAB"], "ravi_patient_modality_match.tsv"), sep="\t", index=False)
    C.log(f"{len(match)} patients; with both MALDI and Visium: {int(match.MALDI_and_Visium.sum())} -> {', '.join(match.patient[match.MALDI_and_Visium])}")

sb = dict(dataset="S-BIAD1426_IDHmut", sample="AREA01_ATO2_SectionA", patient="WEHI IDH-mut (1 section)",
          maldi_file="processed_data/metab_processed.rds (no public .ibd)", polarity="positive", instrument="iMScope QT (AP-MALDI-QTOF)",
          matrix="CHCA", pixel_size_um=10, mz_range="70-400")
f = os.path.join(P["TAB"], "sbiad1426_targets.tsv")
if os.path.exists(f):
    t = pd.read_csv(f, sep="\t")
    for k, m in (("2HG", "2-hydroxyglutarate"), ("lactate", "lactate"), ("GPC", "glycerophosphocholine"), ("NAA", "N-acetylaspartate")):
        sb[f"detect_{k}"] = t.loc[t.metabolite == m, "verdict"].iloc[0] if (t.metabolite == m).any() else ""
master = pd.concat([pd.DataFrame([sb]), maldi], ignore_index=True)
if len(match):
    master = master.merge(match[["patient", "visium_zip_samples", "visium_h5ad_samples", "scRNA_h5ad_samples"]], on="patient", how="left")
master.to_csv(os.path.join(P["TAB"], "master_sample_table.tsv"), sep="\t", index=False)
C.log(f"-> {P['TAB']}/master_sample_table.tsv ({len(master)} rows)")
