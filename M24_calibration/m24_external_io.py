
import glob, gzip, json, os
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio
import p2_common as C


def read_visium(vd):
    mtx = glob.glob(os.path.join(vd, "filtered_feature_bc_matrix", "matrix.mtx*"))[0]
    M = sio.mmread(gzip.open(mtx) if mtx.endswith(".gz") else mtx).tocsc().T.tocsr()
    bc = pd.read_csv(glob.glob(os.path.join(vd, "filtered_feature_bc_matrix", "barcodes.tsv*"))[0], header=None)[0].values
    ft = pd.read_csv(glob.glob(os.path.join(vd, "filtered_feature_bc_matrix", "features.tsv*"))[0], header=None, sep="\t")
    genes = ft[1].astype(str).str.upper().values
    tp = pd.read_csv(os.path.join(vd, "spatial", "tissue_positions.csv")); tp = tp.set_index("barcode").loc[bc]
    sf = json.load(open(os.path.join(vd, "spatial", "scalefactors_json.json")))
    return M, genes, tp[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float), 55.0 / sf["spot_diameter_fullres"]


def binmeans(V, idx, nb):
    """Bin means via a sparse indicator matrix (same result as np.add.at in M23-03)."""
    Bm = sp.csr_matrix((np.ones(len(idx), np.float32), (idx, np.arange(len(idx)))), shape=(nb, len(idx))); n = np.bincount(idx, minlength=nb).astype(np.float32)
    return np.asarray(Bm @ V, dtype=np.float32) / np.maximum(n, 1)[:, None], n


def load_registration(P):
    f = os.path.join(P["PROC"], "m23x_registration.json")
    if not os.path.exists(f):
        raise SystemExit(f"Missing {f}: run M23-03 (Step 22) first")
    return json.load(open(f))


def load_section(name, T, min_det=0.10):
    """T: entry of this section in m23x_registration.json. Returns the pixel matrix (TIC + log1p, features detected in >= min_det of pixels), registered coordinates (µm), checkerboard halves and Visium data."""
    sd = T["dir"]
    Mv, genes, xyv_px, um_per_px = read_visium(os.path.join(sd, "Visium")); xyv = xyv_px * um_per_px
    cf, lf = glob.glob(os.path.join(sd, "MALDI", "cnts_*.tsv"))[0], glob.glob(os.path.join(sd, "MALDI", "locs_*.tsv"))[0]
    Lc = pd.read_csv(lf, sep="\t"); Ct = pd.read_csv(cf, sep="\t", dtype=np.float32, engine="c")
    Ct = Ct.set_index(Ct.columns[0]); Lc = Lc.set_index(Lc.columns[0]).loc[Ct.index]
    X = Ct.values.astype(np.float32); xym = Lc[["x", "y"]].values.astype(float)
    tic = X.sum(1); okp = tic > 0; X, xym, tic = X[okp], xym[okp], tic[okp]
    X = np.log1p(X / tic[:, None] * np.median(tic))
    step = float(np.median(np.diff(np.unique(np.round(xym[:, 0], 3)))))
    gx, gy = np.round((xym[:, 0] - xym[:, 0].min()) / step).astype(int), np.round((xym[:, 1] - xym[:, 1].min()) / step).astype(int); half = (gx + gy) % 2
    R_, cs, ct = np.asarray(T["R"]), np.asarray(T["cs"]), np.asarray(T["ct"])
    xm_um = ((xym - cs) @ R_.T * T["s"] + ct + np.asarray(T["tr"])) * um_per_px
    det = (X > 0).mean(0); keepf = det >= min_det
    return dict(name=name, X=X[:, keepf], fid=np.asarray(Ct.columns)[keepf], xm_um=xm_um, half=half, Mv=Mv, xyv=xyv, px_um=step * T["s"] * um_per_px)


def make_bins(Sx, R, shift=(0.0, 0.0), rot_deg=0.0, n_pc=100, halves=True):
    """Binning as in M23-03. shift / rot_deg: extra translation (µm) / rotation about the centroid (degrees) of the MALDI coordinates.
    Returns dict(Y, A, B (z-scored), PC (<= n_pc gene PCs, not z-scored), xy, folds, im (pixel -> valid bin, -1 if none), n_bins); None if < 60 valid bins."""
    xm = Sx["xm_um"]
    if rot_deg:
        th = np.deg2rad(rot_deg); Rr = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); c0 = xm.mean(0); xm = (xm - c0) @ Rr.T + c0
    xm = xm + np.asarray(shift, float); X, half, Mv, xyv = Sx["X"], Sx["half"], Sx["Mv"], Sx["xyv"]
    o = np.minimum(xm.min(0), xyv.min(0)); km = np.floor((xm - o) / R).astype(int); kv = np.floor((xyv - o) / R).astype(int)
    keym, keyv = km[:, 1] * 100000 + km[:, 0], kv[:, 1] * 100000 + kv[:, 0]
    common = np.intersect1d(keym, keyv); gi = {k: i for i, k in enumerate(common)}; nb = len(common)
    im, iv = np.array([gi.get(k, -1) for k in keym]), np.array([gi.get(k, -1) for k in keyv]); mm_ = im >= 0
    Yf, nf = binmeans(X[mm_], im[mm_], nb); YA, nA = binmeans(X[mm_ & (half == 0)], im[mm_ & (half == 0)], nb); YB, nB = binmeans(X[mm_ & (half == 1)], im[mm_ & (half == 1)], nb)
    vv = iv >= 0; Bv = sp.csr_matrix((np.ones(vv.sum()), (iv[vv], np.where(vv)[0])), shape=(nb, Mv.shape[0])); CB = (Bv @ Mv).tocsr()
    ok = (nA >= 2) & (nB >= 2) & (np.asarray(CB.sum(1)).ravel() > 200)
    if ok.sum() < 60:
        return None
    CB = CB[ok]; tot = np.asarray(CB.sum(1)).ravel(); LG = (sp.diags(1e4 / tot) @ CB).tocsr(); LG.data = np.log1p(LG.data)
    mu = np.asarray(LG.mean(0)).ravel(); var = np.asarray(LG.multiply(LG).mean(0)).ravel() - mu ** 2; hv = np.argsort(-var)[:2000]
    Zg = C.zcols(LG[:, hv].toarray()); U, S_, _ = np.linalg.svd(np.nan_to_num(Zg), full_matrices=False); k = min(n_pc, len(S_)); PC = U[:, :k] * S_[:k]
    xyb = np.column_stack([common % 100000, common // 100000])[ok] * R
    remap = -np.ones(nb, int); remap[np.where(ok)[0]] = np.arange(ok.sum()); im2 = np.where(im >= 0, remap[np.maximum(im, 0)], -1)
    return dict(Y=C.zcols(Yf[ok]), A=C.zcols(YA[ok]), B=C.zcols(YB[ok]), PC=PC, xy=xyb, folds=C.spatial_folds(xyb, 1000, seed=R), im=im2, n_bins=int(ok.sum()))
