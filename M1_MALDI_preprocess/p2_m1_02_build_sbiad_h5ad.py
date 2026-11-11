#!/usr/bin/env python3

import argparse, os, sys
import numpy as np, pandas as pd, scipy.io, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--force", action="store_true")
a = ap.parse_args(); P = C.paths(a.root)
EXP = os.path.join(P["PROC"], "sbiad1426_export")
out_pix = os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad")
out_cell = os.path.join(P["PROC"], "sbiad1426_cells.h5ad")

# author annotation -> coarse lineage (positive controls: white matter = oligodendrocytes, grey matter = neurons)
LINEAGE = {
    "Oligodendrocyte": "Oligodendrocyte", "OPC": "OPC",
    "Excitatory": "Neuron", "Inhibitory": "Neuron",
    "Astrocyte": "Astrocyte",
    "AC-like": "Malignant", "Mixed-AC-like": "Malignant", "OPC-like": "Malignant", "NPC-like": "Malignant",
    "MES-like": "Malignant", "Progenitor": "Malignant",
    "Microglia": "Myeloid", "Activated Microglia": "Myeloid", "Immunosuppressive Microglia": "Myeloid",
    "Macrophage": "Myeloid", "Monocyte": "Myeloid",
    "CD4 T Cell": "Lymphoid", "CD8 T Cell": "Lymphoid", "NK Cell": "Lymphoid", "B Cell": "Lymphoid",
    "Endothelial": "Vascular", "Mural": "Vascular", "Fibroblast": "Vascular",
}

if os.path.exists(out_pix) and not a.force:
    C.log(f"{out_pix} exists (use --force to rebuild)")
else:
    C.log("assembling pixel-level MALDI ...")
    co = pd.read_csv(os.path.join(EXP, "pixels", "coords.tsv.gz"), sep="\t")
    pk = pd.read_csv(os.path.join(EXP, "pixels", "peaks.tsv.gz"), sep="\t")
    M = scipy.io.mmread(os.path.join(EXP, "pixels", "intensity.mtx.gz")).tocsr().astype(np.float32)
    assert M.shape == (len(co), len(pk)), (M.shape, len(co), len(pk))
    obs = pd.DataFrame(dict(x_px=co["x"].values, y_px=co["y"].values, x_um=co["x_new"].values, y_um=co["y_new"].values,
                            empty=co["empty"].astype(bool).values,
                            tic=np.asarray(M.sum(1)).ravel(), n_peaks=np.diff(M.indptr),
                            tic_author=co.get("rowtotals", pd.Series(np.nan, index=co.index)).values),
                       index=co["pixel_id"].astype(str).values)
    var = pd.DataFrame(dict(mz=pk["mz"].values), index=[f"mz{m:.4f}" for m in pk["mz"].values])
    A = ad.AnnData(X=M, obs=obs, var=var)
    A.obsm["spatial"] = obs[["x_um", "y_um"]].values
    A.uns["source"] = dict(dataset="S-BIAD1426 (SpaMTP, Causer et al. Nat Methods 2026; Kriel et al. Cell Rep Methods 2026)",
                           instrument="Shimadzu iMScope QT (AP-MALDI-QTOF)", polarity="positive", pixel_size_um=10,
                           mz_range="70-400", matrix="CHCA (inferred from matrix peaks)",
                           note="no public .ibd; peaks extracted by the data authors, coordinates registered by them to the adjacent Xenium section")
    A.write_h5ad(out_pix, compression="gzip")
    C.log(f"  -> {out_pix}  {A.shape}, non-empty pixels {(~obs['empty']).sum():,}")

if os.path.exists(out_cell) and not a.force:
    C.log(f"{out_cell} exists (use --force to rebuild)")
else:
    C.log("assembling cell-level Xenium + metabolites ...")
    cd = pd.read_csv(os.path.join(EXP, "cells", "coldata.tsv.gz"), sep="\t", low_memory=False)
    genes = pd.read_csv(os.path.join(EXP, "cells", "genes.tsv.gz"), sep="\t")["gene"].astype(str).values
    mzs = pd.read_csv(os.path.join(EXP, "cells", "metab_peaks.tsv.gz"), sep="\t")["mz"].values
    G = scipy.io.mmread(os.path.join(EXP, "cells", "counts_genes.mtx.gz")).T.tocsr().astype(np.float32)
    Mm = scipy.io.mmread(os.path.join(EXP, "cells", "counts_metab.mtx.gz")).T.toarray().astype(np.float32)
    cd.index = cd["cell_id"].astype(str).values
    cd["lineage"] = cd["Anno"].map(LINEAGE).fillna("Other")
    cd["n_genes"] = np.diff(G.indptr); cd["n_counts"] = np.asarray(G.sum(1)).ravel()
    cd["metab_tic"] = Mm.sum(1)
    A = ad.AnnData(X=G, obs=cd, var=pd.DataFrame(index=genes))
    A.obsm["metab"] = Mm
    A.obsm["spatial"] = cd[["x_centroid", "y_centroid"]].values
    for nm in ("umap", "pca"):
        f = os.path.join(EXP, "cells", f"{nm}.tsv.gz")
        if os.path.exists(f):
            r = pd.read_csv(f, sep="\t").set_index("cell_id").loc[A.obs_names]
            A.obsm[f"X_{nm}"] = r.values.astype(np.float32)
    A.uns["metab_mz"] = np.asarray(mzs, float)
    A.uns["source"] = dict(dataset="S-BIAD1426 unified_spe.rds", note="metabolite values: MALDI pixels of the adjacent section mapped to Xenium cells by the data authors")
    A.write_h5ad(out_cell, compression="gzip")
    C.log(f"  -> {out_cell}  {A.n_obs:,} cells x {A.n_vars} genes, {Mm.shape[1]} metabolites")
    C.log("  lineage counts: " + ", ".join(f"{k} {v}" for k, v in cd["lineage"].value_counts().items()))
