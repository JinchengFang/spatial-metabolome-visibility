#!/usr/bin/env python3

import argparse, glob, os, re, sys, zipfile, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--dataset", default="ravi", choices=["ravi"])
ap.add_argument("--files", default="", help="imzML paths (comma-separated) instead of --dataset")
ap.add_argument("--out-name", default="custom")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--max-peaks", type=int, default=1500)
ap.add_argument("--snr", type=float, default=5.0)
ap.add_argument("--bin-ppm", type=float, default=2.0, help="bin width of the mean spectrum of processed data (ppm)")
ap.add_argument("--mean-pixels", type=int, default=20000)
ap.add_argument("--min-freq", type=float, default=0.01, help="centroid data: minimum fraction of tissue pixels in which a peak is detected")
ap.add_argument("--min-tol", type=float, default=3.0)
ap.add_argument("--max-tol", type=float, default=50.0)
ap.add_argument("--force", action="store_true")
a = ap.parse_args(); P = C.paths(a.root)


# ----------------------------------------------------------------------------- imzML reading (memmap fast path)
def read_header(path, nbytes=3_000_000):
    with open(path, "rb") as f:
        txt = f.read(nbytes).decode("utf-8", "ignore")
    g = lambda pat: (re.search(pat, txt, re.I | re.S) or [None, None])[1]
    pol = "positive" if re.search(r'name="positive scan"', txt) else ("negative" if re.search(r'name="negative scan"', txt) else "")
    if not pol:
        b = os.path.basename(path).lower()
        pol = "negative" if re.search(r"neg", b) else ("positive" if re.search(r"pos", b) else "unknown")
    mode = "processed" if "IMS:1000031" in txt else ("continuous" if "IMS:1000030" in txt else "unknown")
    stype = "centroid" if "MS:1000127" in txt else ("profile" if "MS:1000128" in txt else "unknown")
    px = g(r'name="pixel size x"[^>]*value="([\d.]+)"') or g(r'name="pixel size[^"]*"[^>]*value="([\d.]+)"')
    blk = re.search(r"<instrumentConfiguration\b.*?</instrumentConfiguration>", txt, re.S)
    instv = re.search(r'<cvParam[^>]*name="([^"]+)"(?:[^>]*value="([^"]*)")?', blk.group(0)) if blk else None
    matrix = g(r'name="matrix solution"[^>]*value="([^"]+)"') or g(r'name="matrix[^"]*"[^>]*value="([^"]+)"')
    lo = g(r'name="scan window lower limit"[^>]*value="([\d.]+)"'); hi = g(r'name="scan window upper limit"[^>]*value="([\d.]+)"')
    return dict(polarity=pol, mode=mode, spectrum_type=stype, pixel_size_um=float(px) if px else None,
                instrument=((instv.group(2) or instv.group(1)) if instv else ""), matrix=matrix or "",
                scan_window=[float(lo), float(hi)] if lo and hi else None)


class FastImzML:
    DT = {"f": np.float32, "d": np.float64, "i": np.int32, "l": np.int64}

    def __init__(self, path):
        from pyimzml.ImzMLParser import ImzMLParser
        ibd = None
        for ext in (".ibd", ".IBD", ".Ibd"):
            c = re.sub(r"\.imzml$", ext, path, flags=re.I)
            if os.path.exists(c):
                ibd = c; break
        if ibd is None:
            raise FileNotFoundError(f"no .ibd found for {path}")
        p = ImzMLParser(path, parse_lib="ElementTree", ibd_file=None)
        self.coords = np.array([c_[:2] for c_ in p.coordinates], dtype=np.int64)
        self.mzo, self.mzl = np.array(p.mzOffsets, np.int64), np.array(p.mzLengths, np.int64)
        self.ino, self.inl = np.array(p.intensityOffsets, np.int64), np.array(p.intensityLengths, np.int64)
        self.mzt, self.int_t = self.DT[p.mzPrecision], self.DT[p.intensityPrecision]
        self.mm = np.memmap(ibd, dtype=np.uint8, mode="r")
        self.n = len(self.coords)
        self.continuous = bool(len(set(self.mzo.tolist()[:1000])) == 1)

    def mz(self, i):
        isz = np.dtype(self.mzt).itemsize
        return np.frombuffer(self.mm[self.mzo[i]: self.mzo[i] + self.mzl[i] * isz], dtype=self.mzt).astype(np.float64)

    def inten(self, i):
        isz = np.dtype(self.int_t).itemsize
        return np.frombuffer(self.mm[self.ino[i]: self.ino[i] + self.inl[i] * isz], dtype=self.int_t).astype(np.float64)


# ----------------------------------------------------------------------------- tissue mask
def tissue_mask(coords, tic, coarse):
    """Tissue mask: binned spectra (TIC-normalized, log) -> PCA -> k-means (k = 2).
    The cluster occupying >= 70% of the border of the acquisition area is off-tissue; if neither cluster occupies the border (data already
    cropped to tissue), all non-zero pixels are tissue. Followed by morphological clean-up (opening, hole filling, removal of components < 0.5%)."""
    from scipy.ndimage import binary_opening, binary_fill_holes, label
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    pos = tic > 0
    x, y = coords[:, 0] - coords[:, 0].min(), coords[:, 1] - coords[:, 1].min()
    m = pos.copy(); how = "all non-zero pixels (no off-tissue area detected)"
    # region exports (e.g. from SCiLS) contain only tissue pixels; k-means would then split sections or tissue types into
    # "background", so pre-cropped data (coverage of the acquisition box < 80%) are used as they are
    coverage = len(coords) / float((x.max() + 1) * (y.max() + 1))
    if coverage < 0.8:
        return pos, f"pixels pre-cropped to tissue (coverage of the acquisition box {coverage:.0%}): all non-zero pixels are tissue"
    if pos.sum() > 50:
        Z = np.log1p(coarse[pos] / np.maximum(coarse[pos].sum(1, keepdims=True), 1e-12) * 1e4)
        pcs = PCA(n_components=min(10, Z.shape[1] - 1), random_state=0).fit_transform(Z - Z.mean(0))
        lab = KMeans(2, n_init=10, random_state=0).fit_predict(pcs)
        border = (x[pos] == x.min()) | (x[pos] == x.max()) | (y[pos] == y.min()) | (y[pos] == y.max())
        if border.sum() >= 10:
            share = [np.mean(lab[border] == k) for k in (0, 1)]
            k_off = int(np.argmax(share))
            if share[k_off] >= 0.7 and np.mean(lab == k_off) < 0.9:
                m = np.zeros_like(pos); m[np.where(pos)[0][lab != k_off]] = True
                how = f"k-means tissue/background split (background cluster occupies {share[k_off]:.0%} of the border)"
    G = np.zeros((y.max() + 1, x.max() + 1), bool); G[y[m], x[m]] = True
    G = binary_fill_holes(binary_opening(G, iterations=1))
    lab2, n = label(G)
    if n > 1:
        sizes = np.bincount(lab2.ravel()); sizes[0] = 0
        G = np.isin(lab2, np.where(sizes >= max(20, 0.005 * sizes.sum()))[0])
    return G[y, x] & pos, how


# ----------------------------------------------------------------------------- peak picking
def pick_peaks(axis, y, profile, snr, max_peaks, min_tol, max_tol):
    from scipy.ndimage import percentile_filter, gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths, savgol_filter
    y = np.asarray(y, float)
    ys = savgol_filter(y, 7, 2) if profile and len(y) > 50 else gaussian_filter1d(y, 1.0)
    W = 501 if len(ys) > 5000 else max(11, len(ys) // 20 * 2 + 1)
    base = percentile_filter(ys, 20, size=W)
    r = ys - base
    seg = np.array_split(np.arange(len(r)), 60)
    noise = np.zeros_like(r)
    for s in seg:
        v = r[s]; v = v[v <= np.percentile(v, 80)]
        noise[s] = 1.4826 * np.median(np.abs(v - np.median(v)))
    floor = 0.01 * np.median(r[r > 0]) if (r > 0).any() else 1e-12
    noise = np.maximum(noise, max(floor, 1e-12))
    pk, prop = find_peaks(r, height=snr * noise)
    if len(pk) == 0:
        return pd.DataFrame(columns=["mz", "height", "snr", "fwhm_ppm", "half_win_ppm"])
    w = peak_widths(r, pk, rel_height=0.5)
    idx = np.arange(len(axis))
    left = np.interp(w[2], idx, axis); right = np.interp(w[3], idx, axis)
    fwhm_ppm = (right - left) / axis[pk] * 1e6
    k = 0.75 if profile else 1.0
    half = np.clip(k * fwhm_ppm, min_tol, max_tol)
    # centroid m/z: intensity-weighted within the FWHM
    cen = np.array([np.average(axis[int(np.floor(l_)):int(np.ceil(r_)) + 1], weights=np.maximum(r[int(np.floor(l_)):int(np.ceil(r_)) + 1], 1e-12))
                    for l_, r_ in zip(w[2], w[3])])
    T = pd.DataFrame(dict(mz=cen, height=r[pk], snr=r[pk] / noise[pk], fwhm_ppm=fwhm_ppm, half_win_ppm=half)).sort_values("height", ascending=False)
    # peak-shape filter: FWHM more than 3-fold away from the median FWHM of the 50 strongest peaks (noise / merged peaks)
    ref = T.fwhm_ppm.head(50).median()
    T = T[(T.fwhm_ppm <= 3 * ref) & (T.fwhm_ppm >= ref / 3)]
    keep, occ = [], []
    for row in T.itertuples():                          # greedy removal of overlaps, strongest peaks first
        lo, hi = row.mz * (1 - row.half_win_ppm * 1e-6), row.mz * (1 + row.half_win_ppm * 1e-6)
        if any(not (hi < o[0] or lo > o[1]) for o in occ):
            continue
        keep.append(row.Index); occ.append((lo, hi))
        if len(keep) >= max_peaks:
            break
    return T.loc[keep].sort_values("mz").reset_index(drop=True)


def process(path, out_dir, dataset):
    import anndata as ad
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    name = re.sub(r"\.imzml$", "", os.path.basename(path), flags=re.I)
    name = re.sub(r"[^\w.+-]+", "_", name)
    out = os.path.join(out_dir, f"{name}.h5ad")
    if os.path.exists(out) and not a.force:
        return f"{name}: exists, skipped"
    hdr = read_header(path)
    R = FastImzML(path)
    C.log(f"[{name}] {R.n:,} pixels; {hdr['polarity']}; {hdr['mode']}/{hdr['spectrum_type']}; pixel {hdr['pixel_size_um']} µm; instrument {hdr['instrument']}")
    # ---- pass 1: m/z range
    mzmin, mzmax = np.inf, -np.inf
    for i in (range(1) if R.continuous else range(R.n)):
        if R.mzl[i]:
            mz = R.mz(i); mzmin = min(mzmin, mz[0]); mzmax = max(mzmax, mz[-1])
    # ---- pass 1 (continued): TIC + binned spectra (300 log-spaced bins, for tissue segmentation)
    NB = 300; lstep = np.log(mzmax / mzmin) / NB
    tic = np.zeros(R.n); npts = np.zeros(R.n, int); coarse = np.zeros((R.n, NB), np.float32)
    if R.continuous:
        ax0 = R.mz(0); cb = np.minimum((np.log(ax0 / mzmin) / lstep).astype(int), NB - 1)
        edges = np.r_[0, np.flatnonzero(np.diff(cb)) + 1]; cb_u = cb[edges]
    for i in range(R.n):
        it = R.inten(i); tic[i] = it.sum(); npts[i] = it.size
        if not it.size:
            continue
        if R.continuous:
            coarse[i, cb_u] = np.add.reduceat(it, edges)
        else:
            mz = R.mz(i); coarse[i] = np.bincount(np.minimum((np.log(mz / mzmin) / lstep).astype(int), NB - 1), weights=it, minlength=NB)
    tissue, how = tissue_mask(R.coords, tic, coarse)
    del coarse
    C.log(f"[{name}] m/z {mzmin:.2f}-{mzmax:.2f}; tissue pixels {tissue.sum():,}/{R.n:,} ({how})")
    rng = np.random.default_rng(0)
    on = np.where(tissue)[0]; off = np.where(~tissue & (tic > 0))[0]
    on_s = rng.choice(on, min(len(on), a.mean_pixels), replace=False) if len(on) else on
    off_s = rng.choice(off, min(len(off), 5000), replace=False) if len(off) else off
    profile = hdr["spectrum_type"] != "centroid"
    # ---- pass 2: mean spectrum
    if R.continuous:
        axis = R.mz(0)
        mean_on = np.zeros(len(axis)); mean_off = np.zeros(len(axis))
        for i in on_s:
            mean_on += R.inten(i)
        for i in off_s:
            mean_off += R.inten(i)
        mean_on /= max(len(on_s), 1); mean_off /= max(len(off_s), 1)
    else:
        step = np.log1p(a.bin_ppm * 1e-6); nb = int(np.log(mzmax / mzmin) / step) + 2
        acc_on = np.zeros(nb); acc_off = np.zeros(nb)
        for acc, sel in ((acc_on, on_s), (acc_off, off_s)):
            bi, bw = [], []
            for k, i in enumerate(sel):                    # merge bincounts every 500 pixels
                mz, it = R.mz(i), R.inten(i)
                if mz.size:
                    bi.append(np.minimum((np.log(mz / mzmin) / step).astype(int), nb - 1)); bw.append(it)
                if bi and (len(bi) >= 500 or k == len(sel) - 1):
                    acc += np.bincount(np.concatenate(bi), weights=np.concatenate(bw), minlength=nb)[:nb]; bi, bw = [], []
        axis = mzmin * np.exp((np.arange(nb) + 0.5) * step)
        mean_on = acc_on / max(len(on_s), 1); mean_off = acc_off / max(len(off_s), 1)
    peaks = pick_peaks(axis, mean_on, profile and R.continuous, a.snr, a.max_peaks * 2, a.min_tol, a.max_tol)
    if peaks.empty:
        raise RuntimeError("no peaks found (check --snr)")
    lo = peaks.mz.values * (1 - peaks.half_win_ppm.values * 1e-6); hi = peaks.mz.values * (1 + peaks.half_win_ppm.values * 1e-6)
    C.log(f"[{name}] {len(peaks)} candidate peaks in the mean spectrum; median FWHM {peaks.fwhm_ppm.median():.1f} ppm; median window +-{peaks.half_win_ppm.median():.1f} ppm")
    # ---- pass 3: integration
    Xm = np.zeros((R.n, len(peaks)), np.float32)
    if R.continuous:
        li, hi_i = np.searchsorted(axis, lo), np.searchsorted(axis, hi, side="right")
    for i in range(R.n):
        it = R.inten(i)
        if not it.size:
            continue
        cs = np.concatenate([[0.0], np.cumsum(it)])
        if R.continuous:
            Xm[i] = cs[hi_i] - cs[li]
        else:
            mz = R.mz(i); Xm[i] = cs[np.searchsorted(mz, hi, side="right")] - cs[np.searchsorted(mz, lo)]
        if (i + 1) % 50000 == 0:
            C.log(f"[{name}]   integrated {i + 1:,}/{R.n:,}")
    freq = (Xm[tissue] > 0).mean(0) if tissue.any() else np.zeros(len(peaks))
    on_mean = Xm[tissue].mean(0) if tissue.any() else np.zeros(len(peaks))
    off_mean = Xm[~tissue & (tic > 0)].mean(0) if (~tissue & (tic > 0)).sum() >= 50 else np.full(len(peaks), np.nan)
    keep = np.ones(len(peaks), bool)
    if not profile or not R.continuous:
        keep &= freq >= a.min_freq
    order = np.argsort(-on_mean * keep)[: min(a.max_peaks, keep.sum())]
    order = np.sort(order)
    Xm = Xm[:, order]; peaks = peaks.iloc[order].reset_index(drop=True)
    var = peaks.assign(freq_tissue=freq[order], mean_tissue=on_mean[order], mean_offtissue=off_mean[order],
                       in_off_ratio=on_mean[order] / np.maximum(off_mean[order], 1e-12))
    var.index = [f"mz{m:.4f}" for m in var.mz]
    var = var[~var.index.duplicated()]
    Xm = Xm[:, :len(var)]
    obs = pd.DataFrame(dict(x_px=R.coords[:, 0], y_px=R.coords[:, 1], tic=tic, n_points=npts, tissue=tissue),
                       index=[f"{name}_{x}_{y}" for x, y in R.coords])
    A = ad.AnnData(X=Xm, obs=obs, var=var)
    px_um = hdr["pixel_size_um"] or 0
    if px_um:
        A.obsm["spatial"] = R.coords.astype(float) * px_um
    A.uns["polarity"] = hdr["polarity"]; A.uns["pixel_size_um"] = px_um
    A.uns["source"] = dict(dataset=dataset, file=path, sample=name, **{k: (v if v is not None else "") for k, v in hdr.items()},
                           mz_min=float(mzmin), mz_max=float(mzmax), tissue_mask=how)
    A.uns["preprocess"] = dict(snr=a.snr, bin_ppm=a.bin_ppm, max_peaks=a.max_peaks, min_freq=a.min_freq,
                               median_fwhm_ppm=float(peaks.fwhm_ppm.median()))
    A.write_h5ad(out, compression="gzip")
    # ---- QC plots
    x, y = R.coords[:, 0] - R.coords[:, 0].min(), R.coords[:, 1] - R.coords[:, 1].min()
    G = np.full((y.max() + 1, x.max() + 1), np.nan); G[y, x] = np.log10(tic + 1)
    Mk = np.zeros_like(G); Mk[y, x] = tissue
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 2.2]))
    axs[0].imshow(G, cmap="magma"); axs[0].set_title(f"{name}\nlog10 TIC"); axs[0].axis("off")
    axs[1].imshow(Mk, cmap="gray"); axs[1].set_title(f"tissue mask ({tissue.sum():,} px)"); axs[1].axis("off")
    axs[2].plot(axis, mean_on, lw=.5, label="tissue"); axs[2].plot(axis, mean_off, lw=.5, alpha=.6, label="off-tissue")
    axs[2].plot(var.mz, np.interp(var.mz, axis, mean_on), "r.", ms=2, label=f"{len(var)} peaks"); axs[2].set_yscale("symlog")
    axs[2].set_xlabel("m/z"); axs[2].legend(fontsize=7); axs[2].set_title(f"{hdr['polarity']} · {hdr['instrument']} · FWHM~{peaks.fwhm_ppm.median():.0f} ppm", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(P["FIG"], f"{dataset}_{name}_msi_qc.png"), dpi=140); plt.close(fig)
    return f"{name}: {A.n_obs:,} pixels (tissue {tissue.sum():,}) x {A.n_vars} peaks -> {out}"


def ravi_files():
    zp = os.path.join(P["RAVI"], "MALDI_1.zip")
    dest = os.path.join(P["RAVI"], "extracted", "MALDI_1")
    if os.path.exists(zp) and not os.path.exists(os.path.join(dest, ".done")):
        C.log(f"extracting {zp} -> {dest} (once)")
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if "__MACOSX" in n or os.path.basename(n).startswith("._") or n.endswith("/"):
                    continue
                z.extract(n, dest)
        open(os.path.join(dest, ".done"), "w").write("ok")
    return sorted(f for f in glob.glob(os.path.join(dest, "**", "*"), recursive=True) if re.search(r"\.imzml$", f, re.I))


def main():
    jobs = []
    if a.files:
        jobs += [(f, a.out_name) for f in a.files.split(",") if f]
    else:
        jobs += [(f, "ravi") for f in ravi_files()]
    if not jobs:
        raise SystemExit("no imzML found (check that MALDI_1.zip is complete)")
    C.log(f"{len(jobs)} imzML files: " + ", ".join(os.path.basename(f) for f, _ in jobs))
    outs = {ds: os.path.join(P["PROC"], f"{ds}_maldi") for _, ds in jobs}
    for d in outs.values():
        os.makedirs(d, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(process, f, outs[ds], ds): f for f, ds in jobs}
        for fu, f in futs.items():
            try:
                C.log("done: " + fu.result())
            except Exception as e:
                C.log(f"FAILED {os.path.basename(f)}: {type(e).__name__}: {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
