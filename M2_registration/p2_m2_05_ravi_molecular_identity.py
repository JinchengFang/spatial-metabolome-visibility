#!/usr/bin/env python3

import argparse, glob, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--candidates", default="UKF275,UKF262,UKF260,UKF259,UKF248,UKF296")
ap.add_argument("--res", type=float, default=100.0, help="registration raster (µm)")
ap.add_argument("--margin", type=float, default=0.10); ap.add_argument("--n-perm", type=int, default=999)
ap.add_argument("--map-radius", type=float, default=50.0); ap.add_argument("--no-write", action="store_true")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step3"); OUT = os.path.join(P["PROC"], "ravi_paired"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7}); t0 = time.time()

secs = {os.path.basename(f)[:-5]: f for f in sorted(glob.glob(os.path.join(P["PROC"], "ravi_maldi_sections", "S*.h5ad")))}
vis = {os.path.basename(f)[:-5]: f for f in sorted(glob.glob(os.path.join(P["PROC"], "ravi_visium", "*.h5ad")))}
if not secs or not vis:
    raise SystemExit("ravi_maldi_sections or ravi_visium missing (run run_step2.sh 1-2 first)")
cand_vis = {}
for c in [c.strip() for c in a.candidates.split(",") if c.strip()]:
    hit = [s for s in vis if s.startswith(c + "_T") and "IDH" not in s] or [s for s in vis if s.startswith(c)]
    if hit:
        cand_vis[c] = sorted(hit, key=len)[0]
CV = list(cand_vis.values())
PA = ("sulfatide/HexCer", "prog_WM_oligodendrocyte"); PB = ("N-acetylaspartate", "prog_Neuron_cortex")


def tic_norm(X):
    t = X.sum(1); return X / np.maximum(t, 1e-12)[:, None] * np.median(t[t > 0])


def transform(pts, flip, theta, s, c_src, t):
    p = pts.copy()
    if flip:
        p[:, 0] = 2 * c_src[0] - p[:, 0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return s * (p - c_src) @ R.T + t


def raster(pts, vals, res, radius=0.0, origin=None):
    o = pts.min(0) - 2 * max(radius, res) if origin is None else origin
    im, _ = C.rasterize(pts[:, 0], pts[:, 1], vals, res, o[0], o[1],
                        (int((pts[:, 1].max() - o[1]) / res) + 4 + int(2 * radius / res), int((pts[:, 0].max() - o[0]) / res) + 4 + int(2 * radius / res)), radius)
    return im, o


def xc(A_, B_):
    return fftconvolve(A_, B_[::-1, ::-1], mode="full")


def masked_ncc(D, S, minN):
    """D: target (Visium) map, S: source (MALDI) map; NaN = no data. Returns NCC and overlap for every translation (same convention as best_shift in M2-03)."""
    Md, Ms = np.isfinite(D).astype(float), np.isfinite(S).astype(float)
    d, s = np.nan_to_num(D), np.nan_to_num(S)
    N = xc(Md, Ms); SD = xc(d, Ms); SS = xc(Md, s); SDS = xc(d, s); SD2 = xc(d * d, Ms); SS2 = xc(Md, s * s)
    Nn = np.maximum(N, 1)
    num = SDS - SD * SS / Nn
    den = np.sqrt(np.maximum(SD2 - SD ** 2 / Nn, 1e-9) * np.maximum(SS2 - SS ** 2 / Nn, 1e-9))
    r = num / den; r[N < minN] = -1
    return r


def search(src_pts, srcvals, dst_imgs, Dor, res, fit, coarse=True, init=None):
    """fit: map pairs (list indices) used for registration. Returns (score, parameters)."""
    c_src = src_pts.mean(0); best = (-2, None)
    minN = 0.6 * np.isfinite(dst_imgs[0]).sum()
    if coarse:
        grid = [(fl, th, s_) for fl in (False, True) for th in np.deg2rad(np.arange(0, 360, 6)) for s_ in (0.9, 0.95, 1.0, 1.05, 1.1)]
    else:
        fl0, th0, s0 = init
        grid = [(fl0, th, s_) for th in th0 + np.deg2rad(np.arange(-5, 5.1, 1)) for s_ in s0 + np.arange(-0.03, 0.031, 0.01)]
    for fl, th, s_ in grid:
        q = transform(src_pts, fl, th, s_, c_src, np.zeros(2))
        sc = None
        for j in fit:
            S, Sor = raster(q, srcvals[j], res)
            S = (S - np.nanmean(S)) / (np.nanstd(S) + 1e-9)
            r = masked_ncc(dst_imgs[j], S, minN)
            sc = r if sc is None else sc[:min(sc.shape[0], r.shape[0]), :min(sc.shape[1], r.shape[1])] + r[:min(sc.shape[0], r.shape[0]), :min(sc.shape[1], r.shape[1])]
        sc = sc / len(fit); i, jj = np.unravel_index(np.argmax(sc), sc.shape)
        if sc[i, jj] > best[0]:
            new_or = Dor + np.array([(jj - (S.shape[1] - 1)) * res, (i - (S.shape[0] - 1)) * res])
            best = (float(sc[i, jj]), dict(flip=bool(fl), theta=float(th), s=float(s_), c_src=c_src.tolist(), t=(new_or - Sor).tolist()))
    return best


def map_to_spots(src_pts, pr, spots_um, radius):
    q = transform(src_pts, pr["flip"], pr["theta"], pr["s"], np.array(pr["c_src"]), np.array(pr["t"]))
    nb = cKDTree(q).query_ball_point(spots_um, r=radius)
    rows_, cols_ = [], []
    for i, js in enumerate(nb):
        rows_ += [i] * len(js); cols_ += js
    W = sp.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(len(spots_um), len(q)))
    n = np.asarray(W.sum(1)).ravel()
    return q, sp.diags(1 / np.maximum(n, 1)) @ W, n


# ------------------------------------------------------------------ input
SEC = {}
for k, f in secs.items():
    A = ad.read_h5ad(f); X = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X); v = A.var
    L = np.log1p(tic_norm(X.astype(float)))
    nm = v.best_name.astype(str); tr = v.tier.astype(str).isin(["A", "B"])
    js_s = np.where(tr & (nm.str.startswith("ST ") | nm.str.startswith("HexCer")))[0]; js_n = np.where(tr & (nm == "N-acetylaspartate"))[0]
    SEC[k] = dict(pts=A.obs[["x_local_um", "y_local_um"]].values.astype(float), feats=[L[:, js_s].mean(1) if len(js_s) else None, L[:, js_n].mean(1) if len(js_n) else None],
                  n_ions=(len(js_s), len(js_n)))
VIS = {}
for v in CV:
    B = ad.read_h5ad(vis[v], backed="r"); sp_um = B.obsm["spatial_um"][:].astype(float); obs = B.obs[[PA[1], PB[1]]].copy() if PA[1] in B.obs and PB[1] in B.obs else None
    imgs, Dor = [], None
    if obs is not None:
        for col in (PA[1], PB[1]):
            im, o = raster(sp_um, obs[col].values, a.res, radius=55.0, origin=None if Dor is None else Dor); Dor = o if Dor is None else Dor
            imgs.append((im - np.nanmean(im)) / (np.nanstd(im) + 1e-9))
    VIS[v] = dict(spots=sp_um, obs=obs, imgs=imgs, Dor=Dor, uns=dict(B.uns), ar=B.obs[["array_row", "array_col"]].values); B.file.close()
bad = [k for k, d in SEC.items() if d["feats"][0] is None or d["feats"][1] is None] + [v for v, d in VIS.items() if d["obs"] is None]
if bad:
    raise SystemExit(f"sulfatide / NAA annotation or Visium programme scores missing for: {bad}; run run_step1.sh 4 and run_step2.sh 1-2 first")
C.log(f"{len(SEC)} MALDI sections (sulfatide / HexCer ions, NAA ions: {[d['n_ions'] for d in SEC.values()]}); {len(VIS)} candidate Visium samples")


def heldout(k, v, pr, test):
    """Held-out test at spot level: Spearman rho and block-permutation P of the test pair (0 = A, 1 = B) after mapping with parameters pr."""
    s_, V_ = SEC[k], VIS[v]
    q, W, n = map_to_spots(s_["pts"], pr, V_["spots"], a.map_radius); ok = n >= 2
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    m = np.asarray(W[ok] @ s_["feats"][test][:, None]).ravel(); g = V_["obs"].iloc[:, test].values[ok]
    ix = V_["ar"][ok, 1] // 2; iy = V_["ar"][ok, 0]
    r, p, _ = C.block_perm_test(m, g, ix, iy, block=5, n_perm=199)
    return float(r[0]), float(p[0]), int(ok.sum())


def job(args):
    k, v, fits = args; out = {}
    for name, fit in fits.items():
        sc, pr = search(SEC[k]["pts"], SEC[k]["feats"], VIS[v]["imgs"], VIS[v]["Dor"], a.res, fit)
        sc, pr = search(SEC[k]["pts"], SEC[k]["feats"], VIS[v]["imgs"], VIS[v]["Dor"], a.res, fit, coarse=False, init=(pr["flip"], pr["theta"], pr["s"]))
        out[name] = (sc, pr)
    return k, v, out


FITS = {"A": [0], "B": [1], "AB": [0, 1]}
from concurrent.futures import ProcessPoolExecutor
jobs = [(k, v, FITS) for k in SEC for v in VIS]
C.log(f"molecular template matching: {len(jobs)} (section, candidate) pairs, 3 registrations each (A / B / A+B), {a.workers} processes ...")
RES_ = {}
if a.workers > 1:
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, v, o in ex.map(job, jobs):
            RES_[(k, v)] = o
else:
    for j in jobs:
        k, v, o = job(j); RES_[(k, v)] = o
rows = []
for (k, v), o in RES_.items():
    rB, pB, nB = heldout(k, v, o["A"][1], 1)        # registered on A, tested on B
    rA, pA, nA = heldout(k, v, o["B"][1], 0)        # registered on B, tested on A
    rows.append(dict(section=k, visium=v, ncc_A=o["A"][0], ncc_B=o["B"][0], ncc_AB=o["AB"][0], heldout_rho_B_given_A=rB, p_B=pB,
                     heldout_rho_A_given_B=rA, p_A=pA, n_spots=min(nA, nB), heldout_score=np.nanmean([rA, rB]),
                     flip=o["AB"][1]["flip"], rotation_deg=np.rad2deg(o["AB"][1]["theta"]) % 360, scale=o["AB"][1]["s"]))
MI = pd.DataFrame(rows); MI.to_csv(os.path.join(TAB, "ravi_molecular_identity_all.tsv"), sep="\t", index=False)
H = MI.pivot(index="section", columns="visium", values="heldout_score").loc[list(SEC), CV]
H.to_csv(os.path.join(TAB, "ravi_molecular_identity_matrix.tsv"), sep="\t")
C.log("held-out score matrix (rows = MALDI sections, columns = candidate Visium samples):\n" + H.round(3).to_string())

# ------------------------------------------------------------------ assignment and confidence
ri, ci = linear_sum_assignment(-np.nan_to_num(H.values, nan=-1))
ID = []
for i, j in zip(ri, ci):
    k, v = H.index[i], H.columns[j]; row = MI[(MI.section == k) & (MI.visium == v)].iloc[0]
    rowmax = H.loc[k].max() == H.loc[k, v]; colmax = H[v].max() == H.loc[k, v]
    second = H.loc[k].drop(v).max(); margin = H.loc[k, v] - second
    ok_ctrl = (row.heldout_rho_B_given_A > 0) and (row.heldout_rho_A_given_B > 0) and (row.p_A < 0.05) and (row.p_B < 0.05)
    conf = "high (blind molecular assignment)" if (rowmax and colmax and margin >= a.margin and ok_ctrl) else ("medium (molecular assignment with small margin or non-significant control)" if rowmax or colmax else "low")
    ID.append(dict(section=k, visium=v, patient=VIS[v]["uns"].get("patient", v.split("_")[0]), heldout_score=H.loc[k, v], margin=margin,
                   row_max=bool(rowmax), col_max=bool(colmax), controls_ok=bool(ok_ctrl), identity_confidence=conf, method="molecular_blind"))
ID = pd.DataFrame(ID)
# comparison with the shape-based assignment of Step 2
f2 = os.path.join(os.path.dirname(TAB), "Step2", "ravi_identity_assignment.tsv")
if os.path.exists(f2):
    S2 = pd.read_csv(f2, sep="\t").set_index("section")
    ID["step2_shape_visium"] = ID.section.map(S2.visium); ID["agrees_with_step2"] = ID.step2_shape_visium == ID.visium
ID.to_csv(os.path.join(TAB, "ravi_identity_final.tsv"), sep="\t", index=False)
for r in ID.itertuples():
    C.log(f"  {r.section} -> {r.patient}: held-out score {r.heldout_score:+.3f}, margin {r.margin:+.3f}; {r.identity_confidence}")
n_hi = int(ID.identity_confidence.str.startswith("high").sum())
C.log(f"high-confidence identities: {n_hi}/{len(ID)}")

# ------------------------------------------------------------------ rewrite paired files (high confidence only)
if not a.no_write and n_hi:
    for r in ID[ID.identity_confidence.str.startswith("high")].itertuples():
        k, v, pid = r.section, r.visium, r.patient
        pr = RES_[(k, v)]["AB"][1]
        B = ad.read_h5ad(vis[v]); Araw = ad.read_h5ad(secs[k]); Xraw = Araw.X.toarray() if sp.issparse(Araw.X) else np.asarray(Araw.X)
        q, W, n = map_to_spots(SEC[k]["pts"], pr, B.obsm["spatial_um"], a.map_radius); ok = n > 0
        B.obsm["metab"] = np.asarray(W @ tic_norm(Xraw.astype(float))).astype(np.float32); B.obsm["maldi_px_map"] = W.tocsr()
        B.obs["n_maldi_px"] = n; B.obs["has_maldi"] = ok
        mv = Araw.var.copy(); mv.index = mv.index.astype(str)
        B.uns["metab_var_names"] = np.asarray(mv.index); B.uns["maldi_section"] = k
        B.uns["transform"] = {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in pr.items()}
        B.uns["registration_method"] = "molecular_masked_NCC (M2-05)"; B.uns["identity_confidence"] = r.identity_confidence
        B.write_h5ad(os.path.join(OUT, f"{pid}_visium.h5ad"), compression="gzip"); mv.to_csv(os.path.join(OUT, f"{pid}_metab_var.tsv"), sep="\t")
        Araw.obs["x_vis_um"] = q[:, 0]; Araw.obs["y_vis_um"] = q[:, 1]; Araw.obsm["spatial_visium_um"] = q
        Araw.uns["patient"] = pid; Araw.uns["visium_sample"] = v
        Araw.write_h5ad(os.path.join(OUT, f"{pid}_maldi.h5ad"), compression="gzip")
        C.log(f"  rewriting ravi_paired/{pid}_*.h5ad (joint molecular registration: flip {pr['flip']}, rotation {np.rad2deg(pr['theta']) % 360:.0f} deg, scale {pr['s']:.2f}; {int(ok.sum())} spots with MALDI)")

# ------------------------------------------------------------------ figures: held-out matrix and identity montage
fig, ax = plt.subplots(figsize=(5, 3.6))
im = ax.imshow(H.values.astype(float), cmap="magma", aspect="auto"); plt.colorbar(im, ax=ax, label="held-out ρ (mean of A|B, B|A)")
ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index); ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=90)
for r in ID.itertuples():
    if r.visium in H.columns:
        ax.add_patch(plt.Rectangle((list(H.columns).index(r.visium) - .5, list(H.index).index(r.section) - .5), 1, 1, fill=False,
                                   ec="c" if r.identity_confidence.startswith("high") else "grey", lw=2))
ax.set_title("GBM identity by molecular template matching (cross held-out)"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "ravi_molecular_identity_matrix.png"), dpi=160); plt.close(fig)
nc = max(len(SEC), len(VIS))
fig, axs = plt.subplots(4, nc, figsize=(2.1 * nc, 8.4), squeeze=False)
for j, (k, d) in enumerate(SEC.items()):
    for i_, (vals, t_) in enumerate(((d["feats"][0], "sulfatide/HexCer"), (d["feats"][1], "NAA"))):
        im_, _ = raster(d["pts"], vals, 100.0); axs[i_, j].imshow(im_, cmap="viridis"); axs[i_, j].set_title(f"{k} MALDI {t_}", fontsize=6)
for j, (v, d) in enumerate(VIS.items()):
    for i_, t_ in enumerate(("oligodendrocyte program", "neuron program")):
        axs[2 + i_, j].imshow(d["imgs"][i_], cmap="viridis"); axs[2 + i_, j].set_title(f"{v}\n{t_}", fontsize=6)
for axx in axs.ravel():
    axx.axis("off")
fig.suptitle("Identity aid: MALDI sections (rows 1–2) vs candidate Visium (rows 3–4); orientation may differ (flip/rotation)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ravi_identity_montage.png"), dpi=130); plt.close(fig)

C.save_json(dict(n_high=n_hi, assignment=ID.to_dict(orient="records"), runtime_s=round(time.time() - t0)), os.path.join(TAB, "ravi_molecular_identity_summary.json"))
C.log(f"done ({time.time() - t0:.0f} s)")
