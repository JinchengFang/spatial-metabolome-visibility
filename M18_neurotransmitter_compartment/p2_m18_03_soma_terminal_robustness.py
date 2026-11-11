#!/usr/bin/env python3

import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--n-rand", type=int, default=499)
ap.add_argument("--shift-step", default="Step13"); ap.add_argument("--step", default="Step14")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
cfg = L.Config(P, min_px_half=0, min_bins=40); rng = np.random.default_rng(0)
SYN = ["TH", "DDC", "SLC6A3", "SLC18A2"]
TARGETS = {"MSN_all": C.MSN_TARGET_GENES, "MSN_D1": ["DRD1", "TAC1", "PDYN"], "MSN_D2": ["DRD2", "PENK", "ADORA2A"]}
SH = {}
fs = os.path.join(P["P2"], "05_tables", a.shift_step, "m11_registration_summary.tsv")
if os.path.exists(fs):
    rs = pd.read_csv(fs, sep="\t"); SH = {r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples() if r.cv_gain > 0.02}
M, ns, _ = C.NEURO_PANEL["dopamine"]; DA_IONS = [m for _, m in C.fmp10_ions(M, ns)]
variants = [dict(name="base")] + [dict(name="shift", shift=True), dict(name="res100", res=100), dict(name="res400", res=400), dict(name="detrended", detrend=True)]
variants += [dict(name=f"ion:{m:.3f}", ions=[m]) for m in DA_IONS] + [dict(name=f"gene:{g}", genes=[g]) for g in SYN]
variants += [dict(name=f"target:{k}", target=k) for k in ("MSN_D1", "MSN_D2")]
SEC = [S for S in L.load_all(cfg, ["SMA"]) if "fmp10" in str(S.get("matrix", "")).lower() and S.get("brain_area") in ("Striatum", "Substantia Nigra")]
if not SEC:
    C.log("no FMP-10 striatum / substantia nigra sections"); sys.exit(0)
import anndata as ad
CACHE, rows = {}, []
for V in variants:
    R = V.get("res", 200)
    for S in SEC:
        key = (S["sample"], R, bool(V.get("shift")))
        if key not in CACHE:
            G = L.make_grid(S, R, cfg)
            if G is None:
                CACHE[key] = None; continue
            sh = SH.get(S["sample"], (0.0, 0.0)) if V.get("shift") else (0.0, 0.0)
            Y0, cnt = L.pixel_means(S, G, None, sh); ok = cnt >= 1
            var = ad.read_h5ad(os.path.join(P["PROC"], "sma", f"{S['msi_section']}_paired.h5ad"), backed="r").var
            CB = G["CB"][np.where(ok)[0]].tocsc(); tot = np.asarray(CB.sum(1)).ravel()
            keepg = np.where(np.asarray((CB > 0).mean(0)).ravel() >= 0.01)[0]                  # genes detected in >= 1% of bins (memory; random sets drawn from these)
            gn_all = np.asarray(S["genes"]); need = [i for i, g in enumerate(gn_all) if g in set(SYN + sum(TARGETS.values(), []))]
            keepg = np.union1d(keepg, need)
            LG = np.log1p((sp.diags(1e4 / np.maximum(tot, 1)) @ CB[:, keepg]).toarray()).astype(np.float32)
            CACHE = {k: v_ for k, v_ in CACHE.items() if k[1] == R}                              # cache the current resolution only
            CACHE[key] = dict(Y=Y0[ok], xy=G["xy"][ok], LG=LG, var=var, genes=gn_all[keepg])
        D = CACHE[key]
        if D is None:
            continue
        mz, lo, hi = D["var"].mz.values, D["var"].win_lo.values, D["var"].win_hi.values
        ions = V.get("ions", DA_IONS); idx = []
        for t in ions:
            j = int(np.argmin(np.abs(mz - t)))
            if abs(mz[j] - t) / t * 1e6 <= max(5, (hi[j] - lo[j]) / 2 / t * 1e6 + 2) and (np.expm1(D["Y"][:, j]) > 0).mean() >= 0.05:
                idx.append(j)
        if not idx:
            continue
        T2 = C.poly2(D["xy"]); prep = (lambda M_: C.detrend(M_, T2)) if V.get("detrend") else (lambda M_: M_)
        y = C.zcols(prep(C.zcols(D["Y"][:, idx]).mean(1)[:, None]))[:, 0]
        gn = D["genes"]; gpos = {g: i for i, g in enumerate(gn)}
        Zg = C.zcols(prep(D["LG"])); Zg = np.nan_to_num(Zg); mexp = D["LG"].mean(0)
        gi = [gpos[g] for g in V.get("genes", SYN) if g in gpos]; ti = [gpos[g] for g in TARGETS[V.get("target", "MSN_all")] if g in gpos]
        if not gi or Zg[:, gi].mean(1).std() < 1e-9:
            rows.append(dict(variant=V["name"], sample=S["sample"], brain_area=S["brain_area"], rho=np.nan, p=np.nan, R2_syn=np.nan, R2_target=np.nan)); continue
        sc = Zg[:, gi].mean(1); rho = float(np.corrcoef(sc, y)[0, 1])
        qb = np.quantile(mexp[mexp > 0], np.linspace(0, 1, 11)[1:-1]); gb = np.digitize(mexp, qb)
        null = np.array([np.corrcoef(Zg[:, [rng.choice(np.where(gb == gb[g])[0]) for g in gi]].mean(1), y)[0, 1] for _ in range(a.n_rand)]); null = null[np.isfinite(null)]
        folds = C.spatial_folds(D["xy"], 1000, seed=R); yy = y[:, None]
        r2s = float(C.r2_columns(yy, C.cv_ridge_gcv(C.zcols(sc[:, None]), yy, folds))[0])
        r2t = float(C.r2_columns(yy, C.cv_ridge_gcv(C.zcols(Zg[:, ti].mean(1)[:, None]), yy, folds))[0]) if ti else np.nan
        rows.append(dict(variant=V["name"], sample=S["sample"], brain_area=S["brain_area"], n_bins=len(y), n_ions=len(idx), rho=rho,
                         p=float((np.sum(null >= rho) + 1) / (len(null) + 1)), R2_syn=r2s, R2_target=r2t))
    C.log(f"  variant {V['name']}: done")
T = pd.DataFrame(rows); T.to_csv(os.path.join(TAB, "m18_robustness.tsv"), sep="\t", index=False)
summ = []
for v, g in T.groupby("variant", sort=False):
    sn, st = g[g.brain_area == "Substantia Nigra"], g[g.brain_area == "Striatum"]
    d54 = bool(len(sn) and len(st) and (sn.p < 0.05).mean() >= 2 / 3 and (st.p < 0.05).mean() <= 1 / 3 and sn.rho.median() > st.rho.median())
    d55 = bool(len(sn) and len(st) and (sn.R2_syn > sn.R2_target).mean() >= 2 / 3 and (st.R2_target > st.R2_syn).mean() >= 2 / 3)
    summ.append(dict(variant=v, n_SN=len(sn), n_STR=len(st), rho_SN=sn.rho.median(), rho_STR=st.rho.median(), sig_SN=(sn.p < 0.05).mean(), sig_STR=(st.p < 0.05).mean(),
                     R2syn_SN=sn.R2_syn.median(), R2tgt_SN=sn.R2_target.median(), R2syn_STR=st.R2_syn.median(), R2tgt_STR=st.R2_target.median(), D54=d54, D55=d55))
SU = pd.DataFrame(summ); SU.to_csv(os.path.join(TAB, "m18_robustness_summary.tsv"), sep="\t", index=False)
v = dict(n_variants=len(SU), frac_D54=float(SU.D54.mean()), frac_D55=float(SU.D55.mean()), robust_D54=bool(SU.D54.mean() >= 0.8), robust_D55=bool(SU.D55.mean() >= 0.8),
         failing_D54=SU[~SU.D54].variant.tolist(), failing_D55=SU[~SU.D55].variant.tolist())
C.save_json(v, os.path.join(TAB, "m18_robustness_verdict.json"))
C.log("robustness:\n" + SU.round(3).to_string(index=False) + "\ndecision: " + json.dumps(v, ensure_ascii=False))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6)); x = np.arange(len(SU))
axs[0].scatter(x - .1, SU.rho_SN, c="#de2d26", s=18, label="substantia nigra"); axs[0].scatter(x + .1, SU.rho_STR, c="#3182bd", s=18, label="striatum")
axs[0].axhline(0, c="grey", lw=.5); axs[0].set_xticks(x); axs[0].set_xticklabels(SU.variant, rotation=70, fontsize=5); axs[0].set_ylabel("median r(DA, synthesis-gene score)"); axs[0].legend(fontsize=6, frameon=False)
axs[0].set_title(f"D54 holds in {100 * v['frac_D54']:.0f}% of {len(SU)} analysis variants", fontsize=7)
axs[1].scatter(x - .1, SU.R2syn_SN - SU.R2tgt_SN, c="#de2d26", s=18, label="SN: R²(synthesis) − R²(target cells)")
axs[1].scatter(x + .1, SU.R2syn_STR - SU.R2tgt_STR, c="#3182bd", s=18, label="striatum: R²(synthesis) − R²(target cells)")
axs[1].axhline(0, c="grey", lw=.5); axs[1].set_xticks(x); axs[1].set_xticklabels(SU.variant, rotation=70, fontsize=5); axs[1].legend(fontsize=6, frameon=False)
axs[1].set_title(f"D55 holds in {100 * v['frac_D55']:.0f}% of variants", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m18_robustness.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
