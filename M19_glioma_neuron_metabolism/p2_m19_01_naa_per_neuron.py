#!/usr/bin/env python3

import argparse, json, os, re, sys, time, warnings
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--res", default="100,200,400"); ap.add_argument("--n-null", type=int, default=999)
ap.add_argument("--n-perm", type=int, default=9999); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--no-shift", action="store_true")
ap.add_argument("--include-uncertain", action="store_true"); ap.add_argument("--step", default="Step14")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7})
cfg = L.Config(P, ion_set="strict", include_uncertain=a.include_uncertain, min_px_half=1, min_bins=60); rng = np.random.default_rng(0)
RES = [int(x) for x in a.res.split(",")]
MALIG = {"AC-like", "OPC-like", "MES-like", "NPC-like", "Progenitor", "Mixed-AC-like", "Malignant"}
NEUR = {"Excitatory", "Inhibitory", "Neuron"}; OLIGO = {"Oligodendrocyte"}
MALIG_PROG = ["AC_like", "OPC_like", "NPC_like", "MES1_inflammatory", "Hypoxia_MES2", "Cycling"]
SHIFT = {}
fs = os.path.join(P["P2"], "05_tables", "Step8", "m11_registration_summary.tsv")
if os.path.exists(fs) and not a.no_shift:
    rs = pd.read_csv(fs, sep="\t"); rs = rs[(rs.res_um == rs.res_um.min()) & (rs.cv_gain > 0.02)]
    SHIFT = {r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples()}


def z(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def ols_beta(y, B, t):
    X = np.column_stack([np.ones(len(y)), B, t]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[-1])


def partial_r(y, B, t):
    X = np.column_stack([np.ones(len(y)), B]); ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]
    return float(np.corrcoef(ry, rt)[0, 1]), ry


def rotation_nn(xy, n):
    ctr = xy.mean(0); tree = cKDTree(xy); out = []
    for th in rng.uniform(np.deg2rad(30), np.deg2rad(330), n):
        c_, s_ = np.cos(th), np.sin(th); out.append(tree.query((xy - ctr) @ np.array([[c_, -s_], [s_, c_]]).T + ctr)[1])
    return out


def spatial_models(S, R, base_fn, kind):
    """One sample / resolution: every metabolite y ~ baseline + malignant fraction; rotation null. Returns (rows, plot data)."""
    G = L.make_grid(S, R, cfg)
    if G is None:
        return [], None
    sh = SHIFT.get(S["sample"], (0.0, 0.0)); Y0, cnt = L.pixel_means(S, G, None, sh)
    info = base_fn(S, G)
    if info is None:
        return [], None
    ok = (cnt >= 1) & info["keep"]
    if ok.sum() < 40:
        return [], None
    xy = G["xy"][ok]; B = info["B"][ok]; tvar = z(info["malig"][ok]); NN = rotation_nn(xy, a.n_null)
    Btr = np.column_stack([B, C.poly2(xy)[:, 1:]])
    names = np.asarray(S["names"]).astype(str); rows = []; plot = None
    for j, nm in enumerate(names):
        y = Y0[ok, j]
        if np.std(y) == 0 or (np.expm1(y) > 0).mean() < 0.2:
            continue
        y = z(y); b = ols_beta(y, B, tvar); pr, ry = partial_r(y, B, tvar); btr = ols_beta(y, Btr, tvar)
        nb = np.array([ols_beta(y, B, tvar[nn]) for nn in NN])
        rows.append(dict(kind=kind, dataset=S["dataset"], sample=S["sample"], res_um=R, metabolite=nm, n_bins=int(ok.sum()), beta_malignant=b, partial_r=pr,
                         beta_with_trend=btr, p_neg=float((np.sum(nb <= b) + 1) / (len(nb) + 1)), p_pos=float((np.sum(nb >= b) + 1) / (len(nb) + 1)),
                         null_sd=float(nb.std())))
        if re.search(r"acetyl.?l?.?aspart(?!yl)|^NAA", nm, re.I) and (R == 200 or plot is None):
            plot = dict(xy=xy, y=y, ry=ry, malig=info["malig"][ok], neur=info["neur"][ok], R=R, sample=S["sample"])
    return rows, plot


def idh_base(S, G):
    if S.get("ctype") is None:
        return None
    ct = np.asarray(S["ctype"]).astype(str)
    kt = np.floor((S["tx_xy"] - G["o"]) / G["R"]).astype(np.int64); it = G["gi"].reindex(kt[:, 1] * 100000 + kt[:, 0]).fillna(-1).astype(int).values
    nb = len(G["keys"]); v = it >= 0
    cnt = lambda m: np.bincount(it[v & m], minlength=nb).astype(float)
    neur, oli, mal, tot = cnt(np.isin(ct, list(NEUR))), cnt(np.isin(ct, list(OLIGO))), cnt(np.isin(ct, list(MALIG))), cnt(np.ones(len(ct), bool))
    ln = z(np.log1p(neur))
    B = np.column_stack([ln, z(ln ** 2), z(ln ** 3), z(np.log1p(oli)), z(np.log1p(tot))])
    return dict(B=B, malig=mal / np.maximum(tot, 1), neur=neur, keep=(neur >= 1) & (tot >= 3))     # bins with neurons only (NAA per neuron)


def gbm_base(S, G):
    LG = None
    try:
        _, LG, _ = L.tx_blocks(S, G["CB"], G["ncell"], None)
    except Exception:
        return None
    sc, _ = C.score_programs(LG, pd.Index(S["genes"]), C.PROGRAMS, min_genes=3)
    if "Neuron_cortex" not in sc or not any(p in sc for p in MALIG_PROG):
        return None
    ne = z(sc["Neuron_cortex"].values); ol = z(sc["WM_oligodendrocyte"].values) if "WM_oligodendrocyte" in sc else np.zeros(len(ne))
    mal = np.column_stack([z(sc[p].values) for p in MALIG_PROG if p in sc]).mean(1)
    B = np.column_stack([ne, z(ne ** 2), z(ne ** 3), ol, z(np.log1p(np.asarray(G["CB"].sum(1)).ravel()))])
    return dict(B=B, malig=mal, neur=ne, keep=np.ones(len(ne), bool))


rows, plots = [], {}
for S in L.load_all(cfg, ["IDHmut"]):
    for R in RES:
        r_, p_ = spatial_models(S, R, idh_base, "IDHmut_cells"); rows += r_
        if p_ is not None and R == 200:
            plots["IDHmut"] = p_
    C.log(f"  IDH-mut {S['sample']}: " + "; ".join(f"{R} µm NAA beta {r['beta_malignant']:+.3f} (p_neg {r['p_neg']:.3f})" for R in RES for r in rows
                                                if r["res_um"] == R and r["kind"] == "IDHmut_cells" and re.search(r"acetyl.?l?.?aspart(?!yl)", r["metabolite"], re.I)))
for S in L.load_all(cfg, ["GBM"]):
    r_, p_ = spatial_models(S, 200, gbm_base, "GBM_programs"); rows += r_
    if p_ is not None:
        plots[f"GBM {S['sample']}"] = p_
SP = pd.DataFrame(rows)
if len(SP):
    for (k, s, R), g in SP.groupby(["kind", "sample", "res_um"]):
        SP.loc[g.index, "q_neg"] = C.bh(g.p_neg.values); SP.loc[g.index, "q_pos"] = C.bh(g.p_pos.values)
    SP.to_csv(os.path.join(TAB, "m19_spatial_models.tsv"), sep="\t", index=False)
    C.log("spatial models (each metabolite ~ neurons / oligodendrocytes / total cells + malignant fraction):\n" + SP.sort_values(["kind", "res_um", "beta_malignant"]).round(3).to_string(index=False))
# CPTAC
CP = []; boot = None
dcp = os.path.join(P["PROC"], "cptac")
if os.path.exists(os.path.join(dcp, "metabolome_lcimputed.parquet")):
    met = pd.read_parquet(os.path.join(dcp, "metabolome_lcimputed.parquet")); prog = pd.read_csv(os.path.join(dcp, "programs_rna.tsv"), sep="\t", index_col=0)
    rna = pd.read_parquet(os.path.join(dcp, "rna.parquet"))
    com = met.index.intersection(prog.index).intersection(rna.index)
    met, prog, rna = met.loc[com], prog.loc[com], rna.loc[com]
    ne = z(prog["Neuron_cortex"]); ol = z(prog["WM_oligodendrocyte"]); mal = z(np.column_stack([z(prog[p]) for p in MALIG_PROG if p in prog]).mean(1))
    B = np.column_stack([ne, z(ne ** 2), ol])
    targets = {f"metab:{c}": met[c].values for c in met.columns if not str(c).startswith("Unknown")}
    for g in ("NAT8L", "ASPA", "RBFOX3", "SNAP25"):
        if g in rna.columns:
            targets[f"rna:{g}"] = np.log2(rna[g].values + 1) if rna[g].max() > 50 else rna[g].values
    okB = np.isfinite(B).all(1) & np.isfinite(mal)
    for nm, y in targets.items():
        y = np.asarray(y, float); f = okB & np.isfinite(y)                      # some genes (e.g. RBFOX3) have missing values
        if f.sum() < 30 or np.std(y[f]) == 0:
            continue
        y = z(y[f]); Bf, mf = B[f], mal[f]
        b = ols_beta(y, Bf, mf); pr, _ = partial_r(y, Bf, mf)
        nb = np.array([ols_beta(y, Bf, rng.permutation(mf)) for _ in range(a.n_perm)])
        CP.append(dict(target=nm, n=int(f.sum()), beta_malignant=b, partial_r=pr, p_neg=float((np.sum(nb <= b) + 1) / (a.n_perm + 1)), p_pos=float((np.sum(nb >= b) + 1) / (a.n_perm + 1))))
    CP = pd.DataFrame(CP); CP["q_neg"] = C.bh(CP.p_neg.values)
    naa_c = [c for c in met.columns if re.search(r"acetyl.?l?.?aspart(?!yl)", str(c), re.I)]
    if naa_c and "NAT8L" in rna.columns:
        yN, yT = z(met[naa_c[0]].values), z(np.log2(rna["NAT8L"].values + 1) if rna["NAT8L"].max() > 50 else rna["NAT8L"].values); diffs = []
        for _ in range(a.n_boot):
            i = rng.integers(0, len(yN), len(yN)); diffs.append(ols_beta(yN[i], B[i], mal[i]) - ols_beta(yT[i], B[i], mal[i]))
        boot = dict(diff=float(ols_beta(yN, B, mal) - ols_beta(yT, B, mal)), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))
    CP.to_csv(os.path.join(TAB, "m19_cptac_models.tsv"), sep="\t", index=False)
    C.log(f"CPTAC ({len(com)} tumours): each metabolite / transcript ~ neuronal programme (quadratic) + oligodendroglial programme + malignant programme:\n" + CP.sort_values("beta_malignant").round(3).to_string(index=False)
          + (f"\ncoefficient difference NAA - NAT8L {boot['diff']:+.3f} (95% interval {boot['lo']:+.3f} to {boot['hi']:+.3f})" if boot else ""))
# decisions
v = {}
isNAA = lambda s: s.str.contains(r"acetyl.?l?.?aspart(?!yl)", case=False, regex=True)
if len(SP):
    s = SP[(SP.kind == "IDHmut_cells") & isNAA(SP.metabolite)]
    if len(s):
        s200 = s[s.res_um == 200]
        v["D57a_IDHmut_NAA"] = bool(len(s200) and (s200.beta_malignant < 0).all() and (s200.p_neg < 0.05).all() and (s.beta_malignant < 0).all())
        v["D57a_detail"] = {int(r.res_um): dict(beta=round(r.beta_malignant, 3), p_neg=round(r.p_neg, 4), beta_with_trend=round(r.beta_with_trend, 3)) for r in s.itertuples()}
    g = SP[(SP.kind == "IDHmut_cells") & SP.metabolite.str.contains("glycerophosphocholine", case=False) & (SP.res_um == 200)]
    v["ctrl_GPC_positive"] = bool(len(g) and (g.beta_malignant > 0).all() and (g.p_pos < 0.05).any()) if len(g) else None
    gb = SP[(SP.kind == "GBM_programs") & isNAA(SP.metabolite)]
    v["D57c_GBM_same_sign"] = {f"{r.sample} {r.metabolite}": dict(beta=round(r.beta_malignant, 3), p_neg=round(r.p_neg, 4)) for r in gb.itertuples()} if len(gb) else None
if len(CP):
    n_ = CP[isNAA(CP.target)]
    v["D57b_CPTAC_NAA"] = bool(len(n_) and (n_.beta_malignant.iloc[0] < 0) and (n_.p_neg.iloc[0] < 0.05)) if len(n_) else None
    v["D57b_detail"] = n_[["target", "beta_malignant", "partial_r", "p_neg"]].round(4).to_dict("records") if len(n_) else None
    t_ = CP[CP.target == "rna:NAT8L"]; v["NAT8L_detail"] = t_[["beta_malignant", "partial_r", "p_neg"]].round(4).to_dict("records") if len(t_) else None
if boot:
    v["D57d_NAA_beyond_NAT8L"] = bool(boot["hi"] < 0); v["D57d_detail"] = boot
C.save_json(v, os.path.join(TAB, "m19_verdict.json")); C.log("decision: " + json.dumps(v, ensure_ascii=False, default=str))
# figure
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
if "IDHmut" in plots:
    p = plots["IDHmut"]; q = pd.qcut(p["malig"], 5, duplicates="drop")
    m = pd.Series(p["ry"]).groupby(q).agg(["mean", "sem"])
    axs[0].errorbar(np.arange(len(m)), m["mean"], yerr=m["sem"], fmt="o-", c="#de2d26")
    axs[0].set_xticks(np.arange(len(m))); axs[0].set_xticklabels([f"{iv.left:.2f}–{iv.right:.2f}" for iv in m.index], rotation=30)
    axs[0].axhline(0, c="grey", lw=.5); axs[0].set_xlabel("local malignant-cell fraction (quintiles)"); axs[0].set_ylabel("NAA residual after neuron / oligo / total cells (z)")
    axs[0].set_title(f"IDH-mut: NAA per neuron vs infiltration ({p['R']} µm)", fontsize=7)
if len(SP):
    s = SP[(SP.kind == "IDHmut_cells") & (SP.res_um == 200)].sort_values("beta_malignant")
    cols = ["#de2d26" if re.search(r"acetyl.?l?.?aspart(?!yl)", n, re.I) else ("#3182bd" if "glycerophospho" in n else "#bdbdbd") for n in s.metabolite]
    axs[1].barh(np.arange(len(s)), s.beta_malignant, color=cols); axs[1].set_yticks(np.arange(len(s))); axs[1].set_yticklabels(s.metabolite, fontsize=5)
    axs[1].axvline(0, c="k", lw=.5); axs[1].set_xlabel("β malignant fraction (adjusted for neurons, oligodendrocytes, cells)"); axs[1].set_title("IDH-mut, all metabolites (200 µm)", fontsize=7)
if len(CP):
    c = CP[CP.target.str.startswith("metab:") | CP.target.isin(["rna:NAT8L", "rna:ASPA"])].sort_values("beta_malignant")
    hl = c[isNAA(c.target) | c.target.isin(["rna:NAT8L", "rna:ASPA"])]
    axs[2].scatter(np.arange(len(c)), c.beta_malignant, s=6, c="#bdbdbd")
    for r in hl.itertuples():
        k = list(c.target).index(r.target); axs[2].scatter(k, r.beta_malignant, s=30, c="#de2d26" if "metab" in r.target else "#fd8d3c"); axs[2].text(k, r.beta_malignant, " " + r.target.split(":")[1], fontsize=6)
    axs[2].axhline(0, c="k", lw=.5); axs[2].set_xlabel("CPTAC targets (ranked)"); axs[2].set_ylabel("β malignant programs | neuron, oligo programs"); axs[2].set_title("CPTAC GBM (n = 75)", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m19_naa_per_neuron.png"), dpi=180); plt.close(fig)
if "IDHmut" in plots:
    p = plots["IDHmut"]; fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax_, val, t_ in ((axs[0], np.log1p(p["neur"]), "neurons per bin (log)"), (axs[1], p["malig"], "malignant fraction"), (axs[2], p["y"], "NAA (z)"), (axs[3], p["ry"], "NAA per neuron (residual)")):
        ax_.scatter(p["xy"][:, 0], p["xy"][:, 1], c=val, s=5, marker="s", cmap="magma"); ax_.set_title(t_, fontsize=7); ax_.invert_yaxis(); ax_.set_aspect("equal"); ax_.set_xticks([]); ax_.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m19_maps_IDHmut.png"), dpi=170); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
