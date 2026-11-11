#!/usr/bin/env python3

import argparse, glob, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="GBM,ccRCC")
ap.add_argument("--gbm-ref", default=None); ap.add_argument("--ccrcc-ref", default=None)
ap.add_argument("--gbm-label", default=None); ap.add_argument("--ccrcc-label", default=None)
ap.add_argument("--max-cells", type=int, default=300); ap.add_argument("--n-markers", type=int, default=40); ap.add_argument("--min-cells", type=int, default=30)
a = ap.parse_args(); P = C.paths(a.root); OUT = os.path.join(P["PROC"], "signatures"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); t0 = time.time()
CAND = {"GBM": ["annotation_level_3", "annotation_level_2", "cell_type"],
        "ccRCC": ["broad_type", "annotation", "cell_type", "celltype", "Cell_type", "summaryDescription", "cell_type_original"]}
REF = {"GBM": a.gbm_ref or (sorted(glob.glob(os.path.join(P["PUB"], "GBmap_CELLxGENE", "Core_GBmap*.h5ad"))) or [None])[0],
       "ccRCC": a.ccrcc_ref or (sorted(glob.glob(os.path.join(P["PUB"], "Li2022_CancerCell_ccRCC_scRNA", "*.h5ad")), key=os.path.getsize, reverse=True) or [None])[0]}
LABEL = {"GBM": a.gbm_label, "ccRCC": a.ccrcc_label}


def build(ds):
    f = REF[ds]
    if not f or not os.path.exists(f):
        C.log(f"[{ds}] no reference data (GBM: Public_datasets/GBmap_CELLxGENE/Core_GBmap*.h5ad; ccRCC: run p2_step6_downloads.sh first); skipped, M7-04 falls back to marker programmes"); return
    A = ad.read_h5ad(f, backed="r")
    C.log(f"[{ds}] reference {os.path.basename(f)}: {A.n_obs} cells x {A.n_vars} genes; obs columns: {list(A.obs.columns)[:40]}")
    lab = LABEL[ds] or next((c for c in CAND[ds] if c in A.obs and 4 <= A.obs[c].nunique() <= 80), None)
    if lab is None:
        cats = [c for c in A.obs.columns if A.obs[c].dtype.name == "category" and 4 <= A.obs[c].nunique() <= 80]
        raise SystemExit(f"[{ds}] no cell-type label column found; set --{ds.lower()}-label. Candidates (4-80 categories): {cats}")
    labs = A.obs[lab].astype(str).values; vc = pd.Series(labs).value_counts()
    keep_t = vc[vc >= a.min_cells].index.tolist()
    C.log(f"[{ds}] label column '{lab}': {len(vc)} types, {len(keep_t)} with >= {a.min_cells} cells -> {dict(vc.head(30))}")
    idx = np.sort(np.concatenate([rng.choice(np.where(labs == t)[0], min(a.max_cells, (labs == t).sum()), replace=False) for t in keep_t]))
    sub = A[idx].to_memory()
    # raw counts: .raw > layers['counts'] > X (expm1 if X looks log-normalized)
    if sub.raw is not None:
        X = sub.raw.X; vn = sub.raw.var
    elif "counts" in sub.layers:
        X = sub.layers["counts"]; vn = sub.var
    else:
        X = sub.X; vn = sub.var
        mx = X.max() if not sp.issparse(X) else X.data.max()
        if mx < 30:
            X = X.expm1() if sp.issparse(X) else np.expm1(X)
    X = sp.csr_matrix(X).astype(np.float64)
    sym = vn["feature_name"].astype(str).values if "feature_name" in vn else (vn["gene_symbols"].astype(str).values if "gene_symbols" in vn else vn.index.astype(str).values)
    cpm = sp.diags(1e6 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X
    lb = labs[idx]
    M = pd.DataFrame({t: np.asarray(cpm[lb == t].mean(0)).ravel() for t in keep_t}, index=sym)
    M = M.groupby(level=0).max()
    mk = []
    for t in keep_t:
        oth = M.drop(columns=t).max(axis=1)
        lfc = np.log2((M[t] + 1) / (oth + 1)); ok = M[t] >= 5
        mk += list(lfc[ok].sort_values(ascending=False).index[: a.n_markers])
    mk = sorted(set(mk)); SIG = M.loc[mk]
    SIG.to_csv(os.path.join(OUT, f"{ds}_signature.tsv"), sep="\t")
    C.save_json(dict(reference=f, label_column=lab, cell_types=keep_t, n_cells_used=int(len(idx)), n_markers=len(mk),
                     counts_source="raw" if sub.raw is not None else ("layers.counts" if "counts" in sub.layers else "X")), os.path.join(OUT, f"{ds}_signature_info.json"))
    C.log(f"[{ds}] signature: {len(mk)} marker genes x {len(keep_t)} cell types -> {OUT}/{ds}_signature.tsv ({time.time() - t0:.0f} s)")
    A.file.close()


for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
    build(ds)
