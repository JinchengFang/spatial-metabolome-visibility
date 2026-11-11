#!/usr/bin/env python3

import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--ion-set", default="strict", choices=["all", "strict", "protonated"])
ap.add_argument("--res", default="100,200,400"); ap.add_argument("--n-null", type=int, default=99); ap.add_argument("--n-power", type=int, default=20)
ap.add_argument("--effects", default="0.005,0.01,0.02,0.05"); ap.add_argument("--att", type=float, default=0.9, help="signal attenuation by adjacent-section deformation (correlation)")
ap.add_argument("--block-um", type=float, default=1000.0); ap.add_argument("--n-pc", type=int, default=30, help="gene matrix reduced to the first n principal components")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step4")
plt.rcParams.update({"font.size": 7}); t0 = time.time(); rng = np.random.default_rng(0)
EFF = [float(e) for e in a.effects.split(",")]
GT = "" if a.ion_set == "all" else a.ion_set + "_"
ALPHAS = np.logspace(-2, 4, 13)


def cvpred(Xf, y, folds):
    yh = np.empty(len(y))
    for f in np.unique(folds):
        te = folds == f
        yh[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xf[~te], y[~te]).predict(Xf[te])
    return yh


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def poly2(u, v):
    u = (u - u.mean()) / (u.std() + 1e-9); v = (v - v.mean()) / (v.std() + 1e-9)
    return np.column_stack([np.ones_like(u), u, v, u * u, v * v, u * v])


def detrend(M, T):
    M = np.asarray(M, float); M = M[:, None] if M.ndim == 1 else M
    return M - T @ np.linalg.lstsq(T, M, rcond=None)[0]


rows, prow = [], []
for R in [int(r) for r in a.res.split(",")]:
    f = os.path.join(P["PROC"], f"m4_grid_IDHmut_{GT}{R}um.parquet")
    if not os.path.exists(f):
        C.log(f"{f} missing (run M4-01 --ion-set {a.ion_set} --step Step4 first); {R} µm skipped"); continue
    G = pd.read_parquet(f); gx, gy = G.gx.values, G.gy.values
    mcols = [c for c in G if c.startswith("m:")]; ccols = [c for c in G if c.startswith("ct:")]
    gcols = [c for c in G if c.startswith("g:") and G[c].std() > 0]; pcols = [c for c in G if c.startswith("prog:")]
    blk = max(3, int(round(a.block_um / R)))
    bid = (gy // blk) * 100000 + (gx // blk); ub = np.unique(bid); bmap = {b: i for i, b in enumerate(ub)}; bix = np.array([bmap[b] for b in bid])
    folds = np.random.default_rng(R).permutation(np.arange(len(ub)) % 5)[bix]
    Xc0 = np.column_stack([G[ccols].values, np.log1p(G.ncell.values)])
    from sklearn.decomposition import PCA
    for level in ("full", "detrended"):
        T = poly2(gx.astype(float), gy.astype(float))
        Xc = Xc0 if level == "full" else detrend(Xc0, T)
        Gm = G[gcols].values if level == "full" else detrend(G[gcols].values, T)
        Gp = PCA(min(a.n_pc, Gm.shape[1]), random_state=0).fit_transform(StandardScaler().fit_transform(Gm))
        Prg = G[pcols].values if level == "full" else detrend(G[pcols].values, T)
        # programmes residualized on composition -> transcriptomic signal beyond composition (source of the power simulation)
        Pres = Prg - np.column_stack([np.ones(len(Prg)), Xc]) @ np.linalg.lstsq(np.column_stack([np.ones(len(Prg)), Xc]), Prg, rcond=None)[0]
        Pres = (Pres - Pres.mean(0)) / (Pres.std(0) + 1e-9)


        def block_perm_rows(Mx):
            """Swap whole blocks of rows between spatial blocks (cyclic filling truncated to the block when block sizes differ)."""
            order = rng.permutation(len(ub)); out = np.empty_like(Mx)
            src_by_block = [np.where(bix == b)[0] for b in range(len(ub))]
            for b in range(len(ub)):
                dst = src_by_block[b]; src = src_by_block[order[b]]
                out[dst] = Mx[np.resize(src, len(dst))]
            return out


        NULLG = [block_perm_rows(Gp) for _ in range(a.n_null)]
        for mc in mcols:
            y = G[mc].values if level == "full" else detrend(G[mc].values, T)[:, 0]
            yc = cvpred(Xc, y, folds); r_c = r2(y, yc)
            Xcg = np.column_stack([Xc, Gp]); r_cg = r2(y, cvpred(Xcg, y, folds)); dr = r_cg - r_c
            nul = np.array([r2(y, cvpred(np.column_stack([Xc, Gn], ), y, folds)) - r_c for Gn in NULLG])
            thr = float(np.percentile(nul, 95))
            rows.append(dict(res_um=R, level=level, metabolite=mc[2:], n_bins=len(y), R2_composition=r_c, R2_comp_genes=r_cg, dR2_genes=dr,
                             null_dR2_median=float(np.median(nul)), null_dR2_p95=thr, p_perm=(np.sum(nul >= dr) + 1) / (len(nul) + 1)))
            # power: add a programme signal beyond composition
            yres = y - yc; vres = np.var(yres)
            for e in EFF:
                hits = 0
                for k in range(a.n_power):
                    s_true = Pres[:, rng.integers(Pres.shape[1])]
                    s_obs_noise = block_perm_rows(s_true[:, None])[:, 0]
                    s_att = a.att * s_true + np.sqrt(max(0, 1 - a.att ** 2)) * s_obs_noise     # the programme in the MALDI section differs from that measured on the Xenium section
                    beta = np.sqrt(e * vres / (1 - e) / np.var(s_att))
                    ys = y + beta * s_att
                    rc_s = r2(ys, cvpred(Xc, ys, folds)); dr_s = r2(ys, cvpred(Xcg, ys, folds)) - rc_s
                    hits += dr_s > thr
                prow.append(dict(res_um=R, level=level, metabolite=mc[2:], injected_partial_R2=e, power=hits / a.n_power))
        d_ = pd.DataFrame([r for r in rows if r["res_um"] == R and r["level"] == level])
        pw = pd.DataFrame([r for r in prow if r["res_um"] == R and r["level"] == level]).groupby("injected_partial_R2").power.median()
        C.log(f"[{R} µm, {level}] median delta R²(genes | composition) {d_.dR2_genes.median():+.3f} (median null 95th percentile {d_.null_dR2_p95.median():+.3f}); "
              f"metabolites with p < 0.05 {int((d_.p_perm < 0.05).sum())}/{len(d_)}; median power: " + ", ".join(f"delta R²={k:g}: {v:.0%}" for k, v in pw.items()))
BC = pd.DataFrame(rows); PW = pd.DataFrame(prow)
if not len(BC):
    raise SystemExit("no bin matrices available")
BC["q"] = BC.groupby(["res_um", "level"]).p_perm.transform(lambda v: C.bh(v.values))
# significant and positive (the null is often negative, so q alone would call delta R² of about 0 significant); "meaningful" = significant and delta R² >= 0.02
BC["sig_positive"] = (BC.q < 0.05) & (BC.dR2_genes > 0); BC["meaningful"] = BC.sig_positive & (BC.dR2_genes >= 0.02)
BC.to_csv(os.path.join(TAB, "m6_beyond_composition.tsv"), sep="\t", index=False); PW.to_csv(os.path.join(TAB, "m6_power.tsv"), sep="\t", index=False)
sig = BC[BC.meaningful]
for r_ in sig.itertuples():
    C.log(f"   meaningful increment beyond composition (q < 0.05 and delta R² >= 0.02): {r_.metabolite} ({r_.res_um} µm, {r_.level}) delta R² = {r_.dR2_genes:+.3f} (q = {r_.q:.2g})")
summ = {}
for (R, lv), d_ in BC.groupby(["res_um", "level"]):
    pw = PW[(PW.res_um == R) & (PW.level == lv)].groupby("injected_partial_R2").power.median()
    summ[f"{R}um_{lv}"] = dict(median_R2_composition=float(d_.R2_composition.median()), median_dR2_genes=float(d_.dR2_genes.median()),
                               median_null_p95=float(d_.null_dR2_p95.median()), n_sig_q05=int(d_.sig_positive.sum()), n_meaningful=int(d_.meaningful.sum()), n=len(d_),
                               max_dR2=float(d_.dR2_genes.max()), sig_metabolites=sorted(d_[d_.meaningful].metabolite), power_median={f"{k:g}": float(v) for k, v in pw.items()})
C.save_json(dict(ion_set=a.ion_set, attenuation=a.att, by_scale=summ, runtime_s=round(time.time() - t0)), os.path.join(TAB, "m6_03_summary.json"))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for lv, mk in (("full", "o"), ("detrended", "s")):
    s_ = BC[BC.level == lv].groupby("res_um")[["dR2_genes", "null_dR2_p95", "R2_composition"]].median()
    axs[0].plot(s_.index, s_.R2_composition, mk + "-", c="#3182bd", label=f"R² composition ({lv})")
    axs[0].plot(s_.index, s_.dR2_genes, mk + "-", c="#de2d26", label=f"ΔR² genes beyond composition ({lv})")
    axs[0].plot(s_.index, s_.null_dR2_p95, mk + ":", c="grey", label=f"null 95th pct ({lv})")
    p_ = PW[PW.level == lv].groupby(["res_um", "injected_partial_R2"]).power.median().unstack()
    for e in p_.columns:
        axs[1].plot(p_.index, p_[e], mk + "-", label=f"injected ΔR²={e:g} ({lv})")
axs[0].set_xscale("log"); axs[0].set_xlabel("grid (µm)"); axs[0].set_ylabel("median across metabolites"); axs[0].axhline(0, c="k", lw=.4)
axs[0].legend(fontsize=5, frameon=False); axs[0].set_title("IDH-mut: what does the transcriptome add beyond composition?")
axs[1].set_xscale("log"); axs[1].set_ylim(0, 1.05); axs[1].set_xlabel("grid (µm)"); axs[1].set_ylabel("power (median across metabolites)")
axs[1].axhline(.8, ls="--", c="grey", lw=.5); axs[1].legend(fontsize=5, frameon=False); axs[1].set_title(f"Power (signal attenuated ρ={a.att} by adjacent section)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m6_beyond_composition_power.png"), dpi=160); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
