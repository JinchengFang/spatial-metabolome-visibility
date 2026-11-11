#!/usr/bin/env python3

import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--sma-dir", default=None); ap.add_argument("--sections", default="")
ap.add_argument("--n-sub", type=int, default=3000); ap.add_argument("--bin-ppm", type=float, default=2.0); ap.add_argument("--max-peaks", type=int, default=1500)
ap.add_argument("--min-win-ppm", type=float, default=7.5); ap.add_argument("--max-win-ppm", type=float, default=60.0); ap.add_argument("--snr", type=float, default=5.0)
ap.add_argument("--mz-min", type=float, default=50.0); ap.add_argument("--mz-max", type=float, default=1500.0); ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step10")
try:
    from pyimzml.ImzMLParser import ImzMLParser
except ImportError:
    raise SystemExit("pyimzml missing: pip install pyimzml")
D = a.sma_dir or os.path.join(P["PUB"], "Vicari2024_SMA_same_section"); OUT = os.path.join(P["PROC"], "sma"); os.makedirs(OUT, exist_ok=True)
META = {  # msi_metadata.xlsx: section -> (matrix, polarity, mouse, brain area)
    "v11l12-038-a1": ("DHB", "positive", "mPD3", "Striatum"), "v11l12-038-b1": ("DHB", "positive", "mPD3", "Striatum"),
    "v11l12-038-d1": ("9AA", "negative", "mPD3", "Striatum"), "v11l12-109-a1": ("FMP10", "positive", "mPD1", "Striatum"),
    "v11l12-109-b1": ("FMP10", "positive", "mPD3", "Striatum"), "v11l12-109-c1": ("FMP10", "positive", "mPD4", "Striatum"),
    "v11t17-085_a1": ("FMP10", "positive", "mPD1", "Substantia Nigra"), "v11t17-085_b1": ("FMP10", "positive", "mPD3", "Substantia Nigra"),
    "v11t17-085_c1": ("FMP10", "positive", "mPD4", "Substantia Nigra")}
META = {**META, **{k.replace("_", "-"): v for k, v in META.items()}}      # the data files use "_" and "-" interchangeably
files = sorted(glob.glob(os.path.join(D, "MSI", "*.imzML")))
want = [s.strip().lower() for s in a.sections.split(",") if s.strip()]
if want:
    files = [f for f in files if os.path.basename(f).lower().replace(".imzml", "") in want]
if not files:
    raise SystemExit(f"no (selected) imzML + ibd in {D}/MSI; run p2_step10_downloads.sh first")
rng = np.random.default_rng(0); rows = []
edges_log = np.arange(np.log(a.mz_min), np.log(a.mz_max), a.bin_ppm * 1e-6); nbin = len(edges_log)


def spectrum(p, i):
    mz, it = p.getspectrum(i); mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    o = np.argsort(mz) if np.any(np.diff(mz) < 0) else None
    return (mz[o], it[o]) if o is not None else (mz, it)


def mean_spectrum(p, n_sub):
    n = len(p.coordinates); sub = rng.choice(n, min(n_sub, n), replace=False); acc = np.zeros(nbin); cnt = 0
    for i in sub:
        mz, it = spectrum(p, i); m = (mz > a.mz_min) & (mz < a.mz_max)
        if m.sum() == 0:
            continue
        b = np.clip(((np.log(mz[m]) - edges_log[0]) / (a.bin_ppm * 1e-6)).astype(int), 0, nbin - 1)
        acc += np.bincount(b, weights=it[m], minlength=nbin); cnt += 1
    return acc / max(cnt, 1), cnt


def pick_peaks(mean):
    from scipy.ndimage import maximum_filter1d, median_filter
    mzc = np.exp(edges_log + a.bin_ppm * 0.5e-6)
    loc = (mean == maximum_filter1d(mean, int(2 * a.min_win_ppm / a.bin_ppm) | 1)) & (mean > 0)
    base = median_filter(mean, size=int(200 / a.bin_ppm) | 1); mad = median_filter(np.abs(mean - base), size=int(200 / a.bin_ppm) | 1) * 1.4826 + 1e-12
    cand = np.where(loc & ((mean - base) / mad > a.snr))[0]
    cand = cand[np.argsort(-mean[cand])][: a.max_peaks * 2]; wins = []
    for c in cand:
        h = mean[c] / 2; lo = c; hi = c; lim = int(a.max_win_ppm / a.bin_ppm)
        while lo > max(0, c - lim) and mean[lo - 1] > h:
            lo -= 1
        while hi < min(nbin - 1, c + lim) and mean[hi + 1] > h:
            hi += 1
        mw = int(np.ceil(a.min_win_ppm / a.bin_ppm)); lo = min(lo, c - mw); hi = max(hi, c + mw)
        l_, h_ = np.exp(edges_log[max(lo, 0)]), np.exp(edges_log[min(hi + 1, nbin - 1)])
        if any(not (h_ <= w[0] or l_ >= w[1]) for w in wins):
            continue
        wins.append((l_, h_, mzc[c], mean[c]))
        if len(wins) >= a.max_peaks:
            break
    return sorted(wins)


# sections of one matrix share one peak list (pooled mean spectrum) -> identical feature names across sections and animals (conservation tests, M13-03)
groups = {}
for f in files:
    sec = os.path.basename(f).replace(".imzML", "").lower()
    if not (os.path.exists(f[:-6] + ".ibd") or os.path.exists(f[:-6] + ".IBD")):
        C.log(f"[{sec}] .ibd missing; skipped"); continue
    groups.setdefault(META.get(sec, ("?",))[0] + "_" + META.get(sec, ("?", "unknown"))[1], []).append((sec, f))
for gname, members in groups.items():
    todo = [(sec, f) for sec, f in members if a.overwrite or not os.path.exists(os.path.join(OUT, f"{sec}_msi.h5ad"))]
    if not todo:
        C.log(f"[{gname}] all sections done; skipped (--overwrite to redo)"); continue
    t0 = time.time(); parsers = {sec: ImzMLParser(f) for sec, f in members}
    acc = np.zeros(nbin); tot = 0
    for sec, p in parsers.items():
        m_, c_ = mean_spectrum(p, max(500, a.n_sub // len(parsers))); acc += m_ * c_; tot += c_
    wins = pick_peaks(acc / max(tot, 1)); lo_ = np.array([w[0] for w in wins]); hi_ = np.array([w[1] for w in wins])
    C.log(f"[{gname}] pooled mean spectrum of {len(parsers)} sections -> {len(wins)} shared peaks (median window {np.median((hi_ - lo_) / ((hi_ + lo_) / 2)) * 1e6:.0f} ppm); {time.time() - t0:.0f} s")
    for sec, f in todo:
        p = parsers[sec]; n = len(p.coordinates); t1 = time.time()
        md = getattr(p, "imzmldict", {}) or {}; px_um = md.get("pixel size x") or md.get("pixel size (x)") or np.nan
        X = np.zeros((n, len(wins)), np.float32); tic = np.zeros(n); npt = np.zeros(n, int)
        for i in range(n):
            mz, it = spectrum(p, i); tic[i] = it.sum(); npt[i] = len(mz)
            if len(mz) == 0:
                continue
            cs = np.concatenate([[0.0], np.cumsum(it)])
            X[i] = cs[np.searchsorted(mz, hi_, "right")] - cs[np.searchsorted(mz, lo_, "left")]
            if i and i % 10000 == 0:
                C.log(f"     {sec}: {i}/{n} pixels ({time.time() - t1:.0f} s)")
        xy = np.array([c[:2] for c in p.coordinates]); mat, pol, mouse, area = META.get(sec, ("?", "unknown", "?", "?"))
        A = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1], tic=tic, n_points=npt), index=[f"{sec}_{i}" for i in range(n)]),
                       var=pd.DataFrame(dict(mz=[w[2] for w in wins], win_lo=lo_, win_hi=hi_, mean_int=[w[3] for w in wins]), index=[f"{w[2]:.5f}" for w in wins]))
        A.uns.update(dict(section=sec, pixel_size_um=float(px_um) if px_um == px_um else np.nan, polarity=pol, matrix=mat, mouse=mouse, brain_area=area,
                          source=f, peak_group=gname))
        fo = os.path.join(OUT, f"{sec}_msi.h5ad"); A.write_h5ad(fo, compression="gzip")
        rows.append(dict(section=sec, peak_group=gname, n_pixels=n, n_peaks=len(wins), pixel_size_um=px_um, matrix=mat, polarity=pol, mouse=mouse,
                         minutes=(time.time() - t1) / 60))
        C.log(f"[{sec}] {n} pixels -> {fo} ({(time.time() - t1) / 60:.1f} min)")
if rows:
    old = os.path.join(TAB, "m13_imzml_summary.tsv")
    df = pd.concat([pd.read_csv(old, sep="\t"), pd.DataFrame(rows)]).drop_duplicates("section", keep="last") if os.path.exists(old) else pd.DataFrame(rows)
    df.to_csv(old, sep="\t", index=False); C.log("\n" + df.to_string(index=False))
