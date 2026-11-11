#!/usr/bin/env python3

import argparse, glob, json, os, re, sys, zipfile
import numpy as np, pandas as pd, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--force", action="store_true")
a = ap.parse_args(); P = C.paths(a.root)
TAB = os.path.join(os.path.dirname(P["TAB"]), "Step2"); os.makedirs(TAB, exist_ok=True)
DEST = os.path.join(P["RAVI"], "extracted", "10XVisium_2"); OUT = os.path.join(P["PROC"], "ravi_visium")
os.makedirs(DEST, exist_ok=True); os.makedirs(OUT, exist_ok=True)
WANT = re.compile(r"(filtered_feature_bc_matrix\.h5|spatial/tissue_positions(_list)?\.csv|spatial/scalefactors_json\.json|"
                  r"spatial/tissue_(hires|lowres)_image\.png|spatial/detected_tissue_image\.jpg|H&E/[^/]+\.(jpg|jpeg|png|tif|tiff))$", re.I)
clean = lambda s: re.sub(r"[^\w.-]+", "", s.replace("#", ""))

# ---------------------------------------------------------------- 1. extraction
zp = os.path.join(P["RAVI"], "10XVisium_2.zip")
if not os.path.exists(os.path.join(DEST, ".done")):
    C.log(f"extracting the required files {zp} -> {DEST}")
    with zipfile.ZipFile(zp) as z:
        n = 0
        for m in z.namelist():
            if "__MACOSX" in m or os.path.basename(m).startswith("._") or m.endswith("/") or not WANT.search(m):
                continue
            parts = m.split("/")
            sidx = next((i for i, p_ in enumerate(parts) if re.search(r"UKF", p_, re.I)), None)
            if sidx is None:
                continue
            rel = os.path.join(clean(parts[sidx]), *parts[sidx + 1:])
            out = os.path.join(DEST, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                dst.write(src.read())
            n += 1
    open(os.path.join(DEST, ".done"), "w").write("ok"); C.log(f"  {n} files extracted")

# ---------------------------------------------------------------- 2-3. per sample
import scanpy as sc
rows = []
for sdir in sorted(d for d in glob.glob(os.path.join(DEST, "*")) if os.path.isdir(d)):
    s = os.path.basename(sdir)
    h5 = glob.glob(os.path.join(sdir, "outs", "filtered_feature_bc_matrix.h5"))
    pos = glob.glob(os.path.join(sdir, "outs", "spatial", "tissue_positions*.csv"))
    sf = glob.glob(os.path.join(sdir, "outs", "spatial", "scalefactors_json.json"))
    out = os.path.join(OUT, f"{s}.h5ad")
    m_ = re.search(r"UKF(\d+)_?([A-Za-z]+)?", s); pid = f"UKF{int(m_.group(1))}" if m_ else s
    region = re.sub(r"^UKF\d+_?", "", s).replace("_ST", "")
    if not (h5 and pos and sf):
        C.log(f"  [skipped] {s}: h5 / coordinates / scale factors missing"); continue
    if os.path.exists(out) and not a.force:
        B = ad.read_h5ad(out, backed="r")
        rows.append(dict(sample=s, patient=pid, region=region, n_spots=B.n_obs, n_genes=B.n_vars, um_per_px=B.uns["um_per_px"]))
        B.file.close(); continue
    A = sc.read_10x_h5(h5[0]); A.var_names_make_unique()
    p_ = pd.read_csv(pos[0], header=None if pos[0].endswith("_list.csv") else 0)
    p_.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    p_ = p_.set_index("barcode")
    A = A[A.obs_names.isin(p_.index[p_.in_tissue == 1])].copy()
    A.obs = A.obs.join(p_[["array_row", "array_col", "pxl_row", "pxl_col"]])
    # µm per full-resolution pixel: adjacent spots in a row (array_col difference 2) are 100 µm apart
    key = {(r, c): (y, x) for r, c, y, x in zip(p_.array_row, p_.array_col, p_.pxl_row, p_.pxl_col)}
    d = [np.hypot(key[(r, c + 2)][0] - y, key[(r, c + 2)][1] - x) for (r, c), (y, x) in key.items() if (r, c + 2) in key]
    um_per_px = 100.0 / float(np.median(d))
    A.obs["x_um"] = A.obs.pxl_col * um_per_px; A.obs["y_um"] = A.obs.pxl_row * um_per_px
    A.obsm["spatial"] = A.obs[["pxl_col", "pxl_row"]].values.astype(float)       # full-resolution pixels (scanpy / squidpy convention)
    A.obsm["spatial_um"] = A.obs[["x_um", "y_um"]].values.astype(float)
    A.layers["counts"] = A.X.copy()
    A.obs["total_counts"] = np.asarray(A.X.sum(1)).ravel(); A.obs["n_genes"] = np.asarray((A.X > 0).sum(1)).ravel()
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    for progs, pre in ((C.PROGRAMS, "prog_"), (C.METAB_PROGRAMS, "mprog_")):
        S, used = C.score_programs(A.X, A.var_names, progs)
        for c in S:
            A.obs[pre + c] = S[c].values
        A.uns[pre + "genes_used"] = {k: ",".join(v) for k, v in used.items()}
    sfj = json.load(open(sf[0]))
    he = glob.glob(os.path.join(sdir, "H&E", "*")) + glob.glob(os.path.join(sdir, "HE", "*"))
    A.uns.update(dict(sample=s, patient=pid, region=region, um_per_px=um_per_px, scalefactors=sfj,
                      hires_image=os.path.join(sdir, "outs", "spatial", "tissue_hires_image.png"), he_images=he))
    A.write_h5ad(out, compression="gzip")
    rows.append(dict(sample=s, patient=pid, region=region, n_spots=A.n_obs, n_genes=A.n_vars, um_per_px=um_per_px,
                     median_umi=float(np.median(A.obs.total_counts)), median_genes=float(np.median(A.obs.n_genes))))
    C.log(f"  {s}: {A.n_obs} spots, median UMI {np.median(A.obs.total_counts):.0f}, {um_per_px:.3f} µm/px")
T = pd.DataFrame(rows); T.to_csv(os.path.join(TAB, "ravi_visium_samples.tsv"), sep="\t", index=False)
C.log(f"-> {OUT} ({len(T)} samples); {TAB}/ravi_visium_samples.tsv")
