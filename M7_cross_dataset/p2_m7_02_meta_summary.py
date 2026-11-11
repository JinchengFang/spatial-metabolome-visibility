#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--res", type=int, default=200)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step5")
base = os.path.dirname(TAB); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(0)


def boot(v, n=1000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    b = [np.median(rng.choice(v, len(v))) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


rows = []
f6 = os.path.join(base, "Step4", "m6_beyond_composition.tsv")
if os.path.exists(f6):
    B = pd.read_csv(f6, sep="\t")
    for (R, lv), d in B.groupby(["res_um", "level"]):
        adj = d.dR2_genes - d.null_dR2_median
        share = d.R2_composition.clip(lower=0) / (d.R2_composition.clip(lower=0) + adj.clip(lower=0)).clip(lower=1e-6)
        rows.append(dict(dataset="IDH-mut (Xenium 339 genes)", sample="SBIAD1426", res_um=R, level=lv, n=len(d),
                         R2_comp=d.R2_composition.median(), R2_comp_ci=boot(d.R2_composition), dR2_genes=adj.median(), dR2_genes_ci=boot(adj),
                         dR2_genes_raw=d.dR2_genes.median(), dR2_metab=np.nan, share_comp=share.median(),
                         frac_meaningful_genes=float(((d.q < 0.05) & (d.dR2_genes >= 0.02)).mean()), frac_meaningful_metab=np.nan))
f7 = os.path.join(TAB, "m7_spot_decomposition.tsv")
if os.path.exists(f7):
    D = pd.read_csv(f7, sep="\t")
    for (ds, s, R, lv), d in D.groupby(["dataset", "sample", "res_um", "level"]):
        rows.append(dict(dataset=f"{ds} (Visium)", sample=s, res_um=R, level=lv, n=len(d),
                         R2_comp=d.R2_composition.median(), R2_comp_ci=boot(d.R2_composition), dR2_genes=d.dR2_genes_adj.median(), dR2_genes_ci=boot(d.dR2_genes_adj),
                         dR2_genes_raw=d.dR2_genes.median(), dR2_metab=d.dR2_metabprog_adj.median(), share_comp=d.share_composition.median(),
                         frac_meaningful_genes=float(d.meaningful_genes.mean()), frac_meaningful_metab=float(d.meaningful_metabprog.mean()),
                         power_metab_0_05=float(d.get("power_metabprog_0.05", pd.Series(np.nan)).median())))
MB = pd.DataFrame(rows)
if not len(MB):
    raise SystemExit("no results of M6-03 / M7-01")
MB.to_csv(os.path.join(TAB, "m7_meta_by_sample.tsv"), sep="\t", index=False)

summ = {}
for lv in ("full", "detrended"):
    d = MB[(MB.res_um == a.res) & (MB.level == lv)]
    if not len(d):
        continue
    dom = (d.share_comp >= 0.8) & (d.dR2_genes < 0.05)
    dm = d.dropna(subset=["dR2_metab"])
    summ[lv] = dict(n_samples=len(d), n_composition_dominant=int(dom.sum()), D24=bool(dom.mean() >= 2 / 3),
                    median_R2_comp=float(d.R2_comp.median()), median_dR2_genes=float(d.dR2_genes.median()), median_share=float(d.share_comp.median()),
                    D25=bool(len(dm) and (dm.dR2_metab.median() < 0.02) and (dm.frac_meaningful_metab.median() < 0.2)) if len(dm) else None,
                    median_dR2_metab=float(dm.dR2_metab.median()) if len(dm) else None)
if "full" in summ and "detrended" in summ:
    summ["D26"] = summ["full"]["D24"] == summ["detrended"]["D24"]
C.save_json(summ, os.path.join(TAB, "m7_meta_summary.json"))
C.log(json.dumps(summ, ensure_ascii=False, indent=1))

# ------------------------------------------------------------------ forest plot
fig, axs = plt.subplots(1, 2, figsize=(10, 0.28 * max(8, (MB.res_um == a.res).sum() / 2) + 1.5))
for axx, lv in zip(axs, ("full", "detrended")):
    d = MB[(MB.res_um == a.res) & (MB.level == lv)].reset_index(drop=True); yy = np.arange(len(d))[::-1]
    for i, r in d.iterrows():
        lo, hi = r.R2_comp_ci; axx.plot([lo, hi], [yy[i] + .15] * 2, c="#3182bd", lw=1.2); axx.plot(r.R2_comp, yy[i] + .15, "o", c="#3182bd", ms=4)
        lo, hi = r.dR2_genes_ci; axx.plot([lo, hi], [yy[i] - .15] * 2, c="#de2d26", lw=1.2); axx.plot(r.dR2_genes, yy[i] - .15, "s", c="#de2d26", ms=4)
        if np.isfinite(r.dR2_metab):
            axx.plot(r.dR2_metab, yy[i] - .15, "^", c="#fd8d3c", ms=4)
    axx.set_yticks(yy); axx.set_yticklabels([f"{r.dataset} · {r['sample']}" for _, r in d.iterrows()], fontsize=6)
    axx.axvline(0, c="k", lw=.4); axx.set_xlabel("median across structured metabolite features (95% bootstrap CI)")
    axx.set_title(f"{a.res} µm · {lv}", fontsize=8)
fig.suptitle("circles: R² composition; squares: ΔR² whole transcriptome | composition (null-adjusted); triangles: ΔR² metabolic programs | composition", fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, "m7_meta_forest.png"), dpi=170); plt.close(fig)
