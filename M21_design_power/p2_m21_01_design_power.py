#!/usr/bin/env python3

import argparse, itertools, json, os, sys, time
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--n-sim", type=int, default=4000); ap.add_argument("--step", default="Step18")
ap.add_argument("--alpha", type=float, default=0.025, help="two-sided alpha of the primary hypotheses (Bonferroni 0.05/2)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(18)
T = lambda s, f: os.path.join(P["P2"], "05_tables", s, f)
rd = lambda p: pd.read_csv(p, sep="\t") if os.path.exists(p) else None
rj = lambda p: json.load(open(p)) if os.path.exists(p) else {}
isNAA = lambda s: s.astype(str).str.contains(r"acetyl.?l?.?aspart(?!yl)", case=False, regex=True)
VC = dict(source={})
# --- se: SD of the IDH-mut NAA rotation null (the malignant / hypoxia field coefficient of H2 is of the same kind)
sp_ = rd(T("Step14", "m19_spatial_models.tsv")); se_main = {100: 0.09, 200: 0.18, 400: 0.23}
if sp_ is not None:
    s = sp_[(sp_.kind == "IDHmut_cells") & isNAA(sp_.metabolite)]
    if len(s):
        se_main = {int(r.res_um): float(r.null_sd) for r in s.itertuples()}; VC["source"]["se_main"] = "Step14 m19_spatial_models.tsv (IDH-mut NAA null_sd)"
    g = sp_[(sp_.kind == "GBM_programs") & isNAA(sp_.metabolite) & sp_.metabolite.str.contains(r"M-H")]
    if len(g):
        VC["se_main_GBM_Visium_median"] = float(g.null_sd.median())
VC["se_main"] = se_main
# --- se of the interaction and beta_neuron of H1
ia = rd(T("Step15", "m19c_glioma_interaction.tsv")); se_int = {100: 0.04, 200: 0.07, 400: 0.09}; beta_neuron = {100: 0.33, 200: 0.59, 400: 0.74}; tau_obs = {}
if ia is not None:
    s = ia[ia.kind == "IDHmut_cells"]
    if len(s):
        se_int = {int(r.res_um): float(r.null_sd) for r in s.itertuples()}; beta_neuron = {int(r.res_um): float(r.beta_neuron) for r in s.itertuples()}
        VC["source"]["se_int"] = "Step15 m19c_glioma_interaction.tsv"
    g = ia[(ia.kind == "GBM_programs") & ia.metabolite.str.contains(r"M-H") & (ia.p_neuron_pos < 0.05)]
    if len(g) >= 2:
        tau_obs["GBM_beta_neuron_sd_trusted"] = float(g.beta_neuron.std(ddof=1)); tau_obs["GBM_trusted_n"] = int(len(g))
st = rd(T("Step17", "m20_strict_targeted.tsv"))
if st is not None:
    d = st[(st.molecule == "dopamine") & (st.brain_area == "Substantia Nigra")]
    if len(d) >= 2:
        tau_obs["SN_dopamine_r_sd"] = float(d.r.std(ddof=1))
    n_ = st[st.molecule == "N-acetylaspartate"]
    if len(n_) >= 2:
        tau_obs["mouse_NAA_r_sd"] = float(n_.r.std(ddof=1))
VC["se_interaction"] = se_int; VC["beta_neuron_IDHmut"] = beta_neuron; VC["tau_observed"] = tau_obs
ratio = rj(T("Step11", "m15_verdict.json")).get("median_ratio_cross_same_adjusted", 0.67); VC["adjacent_retained_R2_ratio"] = ratio; VC["adjacent_effect_attenuation"] = float(np.sqrt(ratio))
cp = rj(T("Step14", "m19_verdict.json")); VC["CPTAC_NAA_beta_malignant"] = (cp.get("D57b_detail") or [{}])[0].get("beta_malignant")
C.save_json(VC, os.path.join(TAB, "m21_variance_components.json")); C.log("variance components: " + json.dumps(VC, ensure_ascii=False))


def power(delta, se, tau, k, n, alpha, nsim=a.n_sim):
    """n patients; k independent regions (sections) per patient; patient-level one-sample t test (two-sided)."""
    b = rng.normal(delta, tau, (nsim, n)) + rng.normal(0, se / np.sqrt(k), (nsim, n))
    t = b.mean(1) / (b.std(1, ddof=1) / np.sqrt(n)); p = 2 * stats.t.sf(np.abs(t), n - 1)
    return float(np.mean((p < alpha) & (np.sign(b.mean(1)) == np.sign(delta))))


rows = []
NS = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
for (hyp, deltas, se_d), R, tau, k, design in itertools.product(
        [("H1_NAA_tracks_neurons", [beta_neuron.get(200, 0.5) * 0.5, beta_neuron.get(200, 0.5)], se_main), ("H2_hypoxia_lowers_NAA_per_neuron", [0.05, 0.10, 0.15, 0.20, 0.30], se_main),
         ("H3_interaction", [0.05, 0.10, 0.15], se_int)], (100, 200, 400), (0.05, 0.10, 0.15), (1, 2, 4), ("same_section", "adjacent_section")):
    se = se_d.get(R, np.nan)
    if not np.isfinite(se):
        continue
    for dl in deltas:
        d_eff = dl * (1.0 if design == "same_section" else np.sqrt(ratio)); al = a.alpha if hyp != "H3_interaction" else 0.05
        for n in NS:
            rows.append(dict(hypothesis=hyp, delta=round(dl, 3), res_um=R, tau=tau, regions_per_patient=k, design=design, n_patients=n, alpha=al,
                             se_section=se, power=power(d_eff, se, tau, k, n, al)))
PG = pd.DataFrame(rows); PG.to_csv(os.path.join(TAB, "m21_power_grid.tsv"), sep="\t", index=False)


def n_needed(g, target=0.8):
    ok = g[g.power >= target].sort_values("n_patients")
    return int(ok.n_patients.iloc[0]) if len(ok) else None


rec = {}
base = dict(res_um=200, tau=0.10, regions_per_patient=2, design="same_section")
for hyp, dl in (("H1_NAA_tracks_neurons", round(beta_neuron.get(200, 0.5) * 0.5, 3)), ("H2_hypoxia_lowers_NAA_per_neuron", 0.15), ("H3_interaction", 0.10)):
    g = PG[(PG.hypothesis == hyp) & (PG.delta == dl)]
    out = {}
    for k_ in (1, 2, 4):
        for des in ("same_section", "adjacent_section"):
            for R in (100, 200, 400):
                gg = g[(g.res_um == R) & (g.tau == 0.10) & (g.regions_per_patient == k_) & (g.design == des)]
                out[f"{des}_{R}um_k{k_}"] = n_needed(gg)
    rec[hyp] = dict(delta=dl, n_for_80pct_power=out, base_case=out.get("same_section_200um_k2"))
rec["note"] = ("Baseline scenario: 200-µm bins, between-patient heterogeneity tau = 0.10, 2 regions per patient (infiltration zone + hypoxic margin), same section; "
               "H2 effect delta = 0.15 s.d. (about 2/3 of the CPTAC patient-level hypoxia programme effect of -0.23, given the larger noise of spatial bins). None = 80% power not reached within 30 patients")
C.save_json(rec, os.path.join(TAB, "m21_recommendation.json")); C.log("recommended sample sizes: " + json.dumps(rec, ensure_ascii=False, indent=1))
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax_, (hyp, dls) in zip(axs, (("H1_NAA_tracks_neurons", None), ("H2_hypoxia_lowers_NAA_per_neuron", [0.10, 0.15, 0.20]), ("H3_interaction", [0.05, 0.10, 0.15]))):
    g = PG[(PG.hypothesis == hyp) & (PG.res_um == 200) & (PG.tau == 0.10) & (PG.regions_per_patient == 2)]
    for j, dl in enumerate(sorted(g.delta.unique()) if dls is None else dls):
        for des, ls in (("same_section", "-"), ("adjacent_section", "--")):
            gg = g[(g.delta == dl) & (g.design == des)].sort_values("n_patients")
            if len(gg):
                ax_.plot(gg.n_patients, gg.power, ls, c=plt.cm.viridis(j / 3), label=f"δ={dl:.2f} {des.split('_')[0]}")
    ax_.axhline(0.8, c="grey", lw=.5); ax_.set_xlabel("patients"); ax_.set_ylabel("power"); ax_.set_ylim(0, 1.02); ax_.legend(fontsize=5, frameon=False)
    ax_.set_title(f"{hyp} (200 µm, τ = 0.10, 2 regions/patient)", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m21_power_curves.png"), dpi=180); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
