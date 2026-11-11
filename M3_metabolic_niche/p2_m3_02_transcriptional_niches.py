#!/usr/bin/env python3

import argparse, glob, os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--datasets", default="IDHmut,GBM")
ap.add_argument("--res", type=float, default=50.0)
ap.add_argument("--sigma-um", type=float, default=100.0)
ap.add_argument("--kmin", type=int, default=3)
ap.add_argument("--kmax", type=int, default=10)
ap.add_argument("--k-IDHmut", type=int, default=0)
ap.add_argument("--k-GBM", type=int, default=0)
ap.add_argument("--step", default="Step2", help="tables / figures go to 05_tables/<step> and 04_figures/<step>")
a = ap.parse_args(); P = C.paths(a.root)
TAB = os.path.join(os.path.dirname(P["TAB"]), a.step); FIG = os.path.join(os.path.dirname(P["FIG"]), a.step)
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7})
rng = np.random.default_rng(0)


def choose_k(Z, samples, gx, gy, forced, tag):
    multi = len(np.unique(samples)) > 1; rows = []
    for k in range(a.kmin, a.kmax + 1):
        ref = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
        st = [ARI(ref, KMeans(k, n_init=10, random_state=b + 1).fit(Z[rng.choice(len(Z), len(Z) // 2, replace=False)]).predict(Z)) for b in range(10)]
        if multi:   # median leave-one-patient-out transfer ARI
            rep = float(np.median([ARI(ref[samples == s_], KMeans(k, n_init=10, random_state=0).fit(Z[samples != s_]).predict(Z[samples == s_])) for s_ in np.unique(samples)]))
        else:
            fold = ((gx // 20) + (gy // 20)) % 2
            rep = ARI(ref[fold == 1], KMeans(k, n_init=10, random_state=0).fit(Z[fold == 0]).predict(Z[fold == 1]))
        rows.append(dict(k=k, stability_ARI=float(np.mean(st)), reproducibility=float(rep)))
        C.log(f"    [{tag}] k={k}: stability {np.mean(st):.3f}, reproducibility {rep:.3f}")
    K = pd.DataFrame(rows)
    if forced:
        return forced, K
    ok = K[(K.stability_ARI >= 0.85) & (K.reproducibility >= (0.8 if multi else 0.85))]
    kk = int(ok.k.max()) if len(ok) else int(K.sort_values("stability_ARI").k.iloc[-1])
    if kk == a.kmax:
        C.log(f"    [note] selected k equals the upper limit {a.kmax}; consider a larger --kmax")
    return kk, K


def name_niches(lab, F, cols, prefix="T"):
    names = {}
    for l_ in np.unique(lab):
        m = F[lab == l_].mean(0) - F.mean(0)
        top = [cols[j].replace("prog_", "").replace("comp_", "") for j in np.argsort(-m)[:2]]
        names[l_] = f"{prefix}{l_ + 1}:{'+'.join(top)}"
    return np.array([names[l_] for l_ in lab])


summary = {}
dsets = [d.strip() for d in a.datasets.split(",") if d.strip()]
if "IDHmut" in dsets:
    C.log("== IDHmut (Xenium cellular neighbourhoods)")
    pix = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad"), backed="r")
    tis = ~pix.obs["empty"].values.astype(bool)
    px_, py_ = pix.obs.x_um.values[tis], pix.obs.y_um.values[tis]; pix.file.close()
    cel = ad.read_h5ad(os.path.join(P["PROC"], "sbiad1426_cells.h5ad"))
    cx, cy = cel.obsm["spatial"][:, 0], cel.obsm["spatial"][:, 1]
    x0, y0 = min(px_.min(), cx.min()), min(py_.min(), cy.min())             # same grid origin as M3-01
    ix = np.floor((cx - x0) / a.res).astype(int); iy = np.floor((cy - y0) / a.res).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    anno = cel.obs["Anno"].astype(str).values; lin = cel.obs["lineage"].astype(str).values
    vc = pd.Series(anno).value_counts()
    ctype = np.where(pd.Series(anno).map(vc).values >= 100, anno, lin)
    types = sorted(np.unique(ctype))
    sig = a.sigma_um / a.res
    ncell = np.zeros((ny, nx)); np.add.at(ncell, (iy, ix), 1)
    occupied = C.nconv(np.where(ncell > 0, 1.0, 0.0), sig) > 0.2
    comp = {}
    for t in types:
        g = np.zeros((ny, nx)); s = ctype == t; np.add.at(g, (iy[s], ix[s]), 1)
        comp[t] = C.nconv(g, sig) / np.maximum(C.nconv(ncell, sig), 1e-9)
    # gene programmes: summed counts per bin -> log CPM -> smoothing
    G = cel.X.tocsr() if sp.issparse(cel.X) else sp.csr_matrix(cel.X)
    lin_ = iy * nx + ix
    Bm = sp.csr_matrix((np.ones(len(lin_)), (lin_, np.arange(len(lin_)))), shape=(nx * ny, len(lin_)))
    GB = np.asarray((Bm @ G).todense()); tot = GB.sum(1, keepdims=True)
    LG = np.log1p(GB / np.maximum(tot, 1) * 1e4)
    gy, gx = np.where(occupied)
    S_prog, used = C.score_programs(LG.reshape(ny, nx, -1)[gy, gx], cel.var_names, {**C.PROGRAMS, **C.METAB_PROGRAMS}, min_genes=2)
    progs = {}
    for c in S_prog:
        im = np.full((ny, nx), np.nan); im[gy, gx] = S_prog[c].values; progs[c] = C.nconv(im, sig)
    Fc = np.column_stack([np.sqrt(np.clip(comp[t][gy, gx], 0, 1)) for t in types])
    lab_k, K = choose_k(Fc, np.array(["SBIAD1426"] * len(gx)), gx, gy, a.k_IDHmut, "IDHmut")
    K.to_csv(os.path.join(TAB, "transcriptional_niche_k_selection_IDHmut.tsv"), sep="\t", index=False)
    lab = KMeans(lab_k, n_init=10, random_state=0).fit_predict(Fc)
    cols = [f"comp_{t}" for t in types]
    names = name_niches(lab, Fc, cols, "CN")
    O = pd.DataFrame(dict(sample="SBIAD1426", ix=gx, iy=gy, x_um=x0 + (gx + .5) * a.res, y_um=y0 + (gy + .5) * a.res, tniche=names))
    for t in types:
        O[f"comp_{t}"] = comp[t][gy, gx]
    for c, im in progs.items():
        O[f"prog_{c}"] = im[gy, gx]
    O.to_csv(os.path.join(P["PROC"], "transcriptional_niches_IDHmut.tsv.gz"), sep="\t", index=False)
    C.log(f"  -> {lab_k} cellular neighbourhoods: " + ", ".join(f"{n} {v:.1%}" for n, v in pd.Series(names).value_counts(normalize=True).items()))
    summary["IDHmut"] = dict(k=lab_k, niches=pd.Series(names).value_counts().to_dict(), programs_used={k: len(v) for k, v in used.items()})
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    codes = pd.Categorical(names); im = np.full((ny, nx), np.nan); im[gy, gx] = codes.codes
    axs[0].imshow(im, cmap="tab10", vmin=-0.5, vmax=9.5, interpolation="nearest"); axs[0].axis("off"); axs[0].set_title("IDHmut cellular neighbourhoods")
    axs[0].legend([plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap("tab10")(i)) for i in range(len(codes.categories))], codes.categories, fontsize=6, loc="lower left")
    Mx = pd.DataFrame(Fc ** 2, columns=types).groupby(names).mean()
    i2 = axs[1].imshow(Mx.values, cmap="viridis", aspect="auto"); axs[1].set_yticks(range(len(Mx))); axs[1].set_yticklabels(Mx.index)
    axs[1].set_xticks(range(len(types))); axs[1].set_xticklabels(types, rotation=90); fig.colorbar(i2, ax=axs[1], label="mean fraction")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "transcriptional_niches_IDHmut.png"), dpi=140); plt.close(fig)

if "GBM" in dsets:
    fs = sorted(glob.glob(os.path.join(P["PROC"], "ravi_paired", "*_visium.h5ad")))
    if not fs:
        C.log("== GBM: no ravi_paired/*_visium.h5ad (run M2-03 first), skipped")
    else:
        C.log(f"== GBM (Visium programme niches, {len(fs)} patients)")
        blocks = []
        for f in fs:
            B = ad.read_h5ad(f)
            pc = [c for c in B.obs.columns if c.startswith("prog_")]
            Fz = B.obs[pc].values; Fz = (Fz - Fz.mean(0)) / (Fz.std(0) + 1e-9)
            blocks.append((B.uns.get("patient", os.path.basename(f).split("_")[0]), B, pc, Fz))
        pc = blocks[0][2]
        Z = np.vstack([b_[3] for b_ in blocks]); samples = np.concatenate([[b_[0]] * b_[1].n_obs for b_ in blocks])
        gx = np.concatenate([b_[1].obs.array_col.values // 2 for b_ in blocks]); gy = np.concatenate([b_[1].obs.array_row.values for b_ in blocks])
        k, K = choose_k(Z, samples, gx, gy, a.k_GBM, "GBM")
        K.to_csv(os.path.join(TAB, "transcriptional_niche_k_selection_GBM.tsv"), sep="\t", index=False)
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
        names = name_niches(lab, Z, pc, "T")
        out = []; i0 = 0
        for sname, B, _, Fz in blocks:
            n = B.n_obs
            d = pd.DataFrame(dict(sample=sname, barcode=B.obs_names, x_um=B.obsm["spatial_um"][:, 0], y_um=B.obsm["spatial_um"][:, 1],
                                  array_row=B.obs.array_row.values, array_col=B.obs.array_col.values, tniche=names[i0:i0 + n],
                                  has_maldi=B.obs.get("has_maldi", pd.Series(True, index=B.obs_names)).values))
            for c in [c for c in B.obs.columns if c.startswith("prog_") or c.startswith("mprog_")]:
                d[c] = B.obs[c].values
            out.append(d); i0 += n
        O = pd.concat(out, ignore_index=True); O.to_csv(os.path.join(P["PROC"], "transcriptional_niches_GBM.tsv.gz"), sep="\t", index=False)
        tab = pd.crosstab(O.tniche, O["sample"], normalize="columns").round(3)
        tab.to_csv(os.path.join(TAB, "transcriptional_niche_composition_GBM.tsv"), sep="\t")
        C.log(f"  -> k = {k}; share of each niche per patient:\n{tab.to_string()}")
        summary["GBM"] = dict(k=k, niches=O.tniche.value_counts().to_dict())
        nb = len(blocks); nc = min(3, nb); nr = int(np.ceil(nb / nc))
        cats = sorted(O.tniche.unique()); cmap = plt.get_cmap("tab10")
        fig, axs = plt.subplots(nr, nc, figsize=(4.5 * nc, 4 * nr), squeeze=False)
        for axx, (sname, B, _, _) in zip(axs.ravel(), blocks):
            d = O[O["sample"] == sname]
            axx.scatter(d.x_um, d.y_um, c=[cmap(cats.index(t) % 10) for t in d.tniche], s=3); axx.invert_yaxis(); axx.set_aspect("equal"); axx.axis("off"); axx.set_title(sname)
        for axx in axs.ravel()[nb:]:
            axx.axis("off")
        fig.legend([plt.Rectangle((0, 0), 1, 1, color=cmap(i % 10)) for i in range(len(cats))], cats, loc="lower center", ncol=min(4, len(cats)), fontsize=6)
        fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "transcriptional_niches_GBM.png"), dpi=140); plt.close(fig)
C.save_json(summary, os.path.join(TAB, "transcriptional_niches_summary.json"))
