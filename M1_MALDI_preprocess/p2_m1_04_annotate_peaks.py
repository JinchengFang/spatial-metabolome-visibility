#!/usr/bin/env python3

import argparse, glob, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--max-tol", type=float, default=15.0, help="upper limit of the matching tolerance (ppm)")
ap.add_argument("--default-tol", type=float, default=10.0, help="tolerance used when there are too few anchors")
ap.add_argument("--corr-bin-um", type=float, default=100.0, help="bin size (µm) for image correlations")
ap.add_argument("--files", default="")
a = ap.parse_args(); P = C.paths(a.root)

TARGETS = ["2-hydroxyglutarate", "lactate", "glutamate", "glutamine", "alpha-ketoglutarate", "glycerophosphocholine",
           "phosphocholine", "choline", "creatine", "N-acetylaspartate", "taurine", "hexose", "hexose-phosphate",
           "succinate", "citrate", "glutathione", "adenosine", "kynurenine", "tryptophan", "spermidine", "carnitine",
           "acetylcarnitine", "heme b", "cholesterol", "ST 42:2;O2 (sulfatide 24:1)", "HexCer 42:2;O2", "PC 34:1",
           "CE 18:1", "FA 22:6", "PI 38:4"]


def external_db(mz_lo, mz_hi):
    """Optional external library; parsed result cached as tsv.gz."""
    out = []
    for src, pats in (("HMDB", ["hmdb*structures*.zip", "structures.zip", "structures.sdf", "hmdb*.sdf"]),
                      ("LIPIDMAPS", ["LMSD*.zip", "LMSD*.sdf", "lmsd*.sdf"])):
        files = [f for p in pats for f in glob.glob(os.path.join(P["REFDB"], p))]
        if not files:
            continue
        cache = os.path.join(P["REFDB"], f"_parsed_{src}.tsv.gz")
        if os.path.exists(cache):
            t = pd.read_csv(cache, sep="\t")
        else:
            C.log(f"  parsing external library {files[0]} (once) ...")
            t = C.load_external_sdf(files[0], src, 0, 3000, max_mass=3000)
            t.to_csv(cache, sep="\t", index=False)
        out.append(t[(t.mono_mass > mz_lo - 80) & (t.mono_mass < mz_hi + 2)])
    return pd.concat(out) if out else None


def ext_ions(db, polarity):
    if db is None or db.empty:
        return None
    ads = ({"[M+H]+": C.PROTON, "[M+Na]+": C.ELEM["Na"] - C.E_MASS, "[M+K]+": C.ELEM["K"] - C.E_MASS}
           if polarity == "positive" else {"[M-H]-": -C.PROTON, "[M+Cl]-": C.ELEM["Cl"] + C.E_MASS})
    rows = [pd.DataFrame(dict(name=db["source"] + ":" + db["name"].astype(str).str.slice(0, 60), formula=db["formula"],
                              adduct=k, mz=db["mono_mass"] + v, cls=db["source"], relevance="", anchor=False,
                              is_matrix=False, m1_ratio=np.nan)) for k, v in ads.items()]
    return pd.concat(rows).sort_values("mz").reset_index(drop=True)


def binned_images(A, nbin):
    """TIC-normalize and pool pixels into nbin x nbin bins. Returns (bin x peak) means, bin coordinates, tissue pixels per bin."""
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    tic = np.asarray(X.sum(1)).ravel()
    tissue = A.obs["tissue"].values.astype(bool) if "tissue" in A.obs else tic > 0
    keep = tissue & (tic > 0)
    s = np.median(tic[keep]) / np.maximum(tic, 1e-9)
    Xn = sp.diags(np.where(keep, s, 0.0)) @ X
    x = A.obs["x_px"].values.astype(int); y = A.obs["y_px"].values.astype(int)
    ix = (x - x.min()) // nbin; iy = (y - y.min()) // nbin
    lin = iy * (ix.max() + 1) + ix
    u, inv = np.unique(lin[keep], return_inverse=True)
    Bm = sp.csr_matrix((np.ones(keep.sum()), (inv, np.where(keep)[0])), shape=(len(u), A.n_obs))
    cnt = np.asarray(Bm.sum(1)).ravel()
    img = np.asarray((Bm @ Xn).todense()) / cnt[:, None]
    gx = u % (ix.max() + 1); gy = u // (ix.max() + 1)
    good = cnt >= max(2, nbin * nbin // 3)                 # bins with at least one third of their pixels in tissue
    return img[good], gx[good], gy[good], Xn, keep


def corr_cols(M, i, js):
    a = np.log1p(M[:, i]); out = []
    for j in js:
        b = np.log1p(M[:, j])
        out.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
    return np.array(out)


def annotate(path):
    A = ad.read_h5ad(path)
    name = os.path.basename(path).replace(".h5ad", "").replace("_maldi_pixels", "")
    src = A.uns.get("source", {}) if isinstance(A.uns.get("source", {}), dict) else {}
    pol = str(A.uns.get("polarity", src.get("polarity", "positive"))).lower()
    pol = "negative" if pol.startswith("neg") else "positive"
    if "tissue" not in A.obs:
        A.obs["tissue"] = ~A.obs["empty"].astype(bool) if "empty" in A.obs else True
    mz = A.var["mz"].values.astype(float)
    C.log(f"== {name}: {A.n_obs:,} pixels x {A.n_vars} peaks, {pol}, m/z {mz.min():.1f}-{mz.max():.1f}")
    ions = C.build_ion_table(pol, (mz.min(), mz.max()))
    mz_cal, crep, anc = C.calibrate_mass(mz, ions, search_ppm=25)
    tol = float(np.clip(3 * crep["mad_ppm"], 3, a.max_tol)) if crep["applied"] and np.isfinite(crep["mad_ppm"]) else a.default_tol
    crep.update(tolerance_ppm=tol, dataset=name, polarity=pol, n_peaks=int(A.n_vars))
    C.log(f"  mass calibration: {crep['n_anchor_hits']} anchors, median error {crep['median_ppm']:.2f} ppm, "
          f"slope {crep['slope_ppm_per_mz']:.4f} ppm per m/z, residual MAD {crep['mad_ppm']:.2f} ppm -> tolerance +-{tol:.1f} ppm")
    m = C.match_peaks(mz_cal, ions, tol)
    ext = ext_ions(external_db(mz.min(), mz.max()), pol)
    m_ext = C.match_peaks(mz_cal, ext, tol) if ext is not None else pd.DataFrame()

    # ---- image statistics
    px_um = float(A.uns.get("pixel_size_um", src.get("pixel_size_um", 0)) or 0)
    nbin = max(1, int(round(a.corr_bin_um / px_um))) if px_um > 0 else 2
    img, gx, gy, Xn, keep = binned_images(A, nbin)
    L = np.log1p(img)
    ok_cols = (np.asarray((Xn[keep] > 0).mean(0)).ravel() >= 0.01) & (L.std(0) > 0)
    R = np.corrcoef(L[:, ok_cols].T); r95 = float(np.nanquantile(R[np.triu_indices_from(R, 1)], 0.95)) if ok_cols.sum() > 5 else 0.5
    thr = max(0.3, r95)
    crep.update(corr_bin_px=nbin, pixel_size_um=px_um, null_pair_r95=r95, corroboration_threshold=thr)
    C.log(f"  bins of {nbin}x{nbin} pixels ({'about %.0f µm' % (nbin * px_um) if px_um else 'pixel size unknown'}); random-pair correlation r95 = {r95:.2f} -> threshold r >= {thr:.2f}")
    Xb = (Xn[keep] > 0)
    frac = np.asarray(Xb.mean(0)).ravel()
    mean_int = np.asarray(Xn[keep].mean(0)).ravel()
    grid = np.full((gy.max() + 1, gx.max() + 1), np.nan)
    moran = []
    for j in range(A.n_vars):
        grid[:] = np.nan; grid[gy, gx] = np.log1p(img[:, j]); moran.append(C.morans_i_grid(grid))
    moran = np.array(moran)
    # background ratio (if there are off-tissue pixels)
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    off = (~A.obs["tissue"].values.astype(bool)) & (np.asarray(X.sum(1)).ravel() > 0)
    premasked = "pre-cropped to tissue" in str(src.get("tissue_mask", ""))
    if off.sum() >= 100 and not premasked:
        on_mean = np.asarray(X[keep].mean(0)).ravel(); off_mean = np.asarray(X[off].mean(0)).ravel()
        bg_ratio = on_mean / np.maximum(off_mean, 1e-9)
    else:
        bg_ratio = np.full(A.n_vars, np.nan)

    # ---- isotopes: is a peak the 13C M+1 of a stronger peak?
    is_iso_of = np.full(A.n_vars, -1)
    for i in range(A.n_vars):
        t = mz_cal[i] + C.C13_DELTA; w = t * tol * 1e-6
        js = np.where(np.abs(mz_cal - t) <= w)[0]
        for j in js:
            if mean_int[j] < mean_int[i] and corr_cols(img, i, [j])[0] >= max(thr, 0.5):
                is_iso_of[j] = i

    # ---- per-peak summary
    rows = []
    for i in range(A.n_vars):
        cand = m[m.peak_idx == i].sort_values("ppm", key=np.abs) if not m.empty else pd.DataFrame()
        cand_bio = cand[~cand.is_matrix] if len(cand) else cand
        cext = m_ext[m_ext.peak_idx == i].sort_values("ppm", key=np.abs) if len(m_ext) else pd.DataFrame()
        r = dict(peak=A.var_names[i], mz=mz[i], mz_cal=mz_cal[i], frac_pixels=frac[i], mean_int_ticnorm=mean_int[i],
                 morans_i=moran[i], bg_ratio=bg_ratio[i], isotope_of=(A.var_names[is_iso_of[i]] if is_iso_of[i] >= 0 else ""),
                 best_name="", best_formula="", best_adduct="", best_ppm=np.nan, best_class="", relevance="",
                 n_candidates=0, ambiguous=False, candidates="", adduct_partners="", adduct_corr_max=np.nan,
                 iso_peak="", iso_corr=np.nan, iso_ratio_obs_over_theo=np.nan, ext_candidates="")
        if len(cand):
            b = cand_bio.iloc[0] if len(cand_bio) else cand.iloc[0]
            r.update(best_name=b["name"], best_formula=b.formula, best_adduct=b.adduct, best_ppm=b.ppm, best_class=b.cls,
                     relevance=b.relevance, n_candidates=cand["name"].nunique(), ambiguous=cand_bio["name"].nunique() > 1 if len(cand_bio) else False,
                     candidates="; ".join(f"{x.name}{x.adduct}({x.ppm:+.1f})" for x in cand.itertuples()))
            # adduct co-localization: matches of the same metabolite on other peaks
            if not b.is_matrix:
                part = m[(m["name"] == b["name"]) & (m.peak_idx != i)]
                if len(part):
                    rr = corr_cols(img, i, part.peak_idx.values)
                    r.update(adduct_partners="; ".join(f"{ad_}@{mz_cal[j]:.4f}(r={c:.2f})" for ad_, j, c in zip(part.adduct, part.peak_idx, rr)),
                             adduct_corr_max=np.nanmax(rr) if np.isfinite(rr).any() else np.nan)
                # 13C isotope
                t = mz_cal[i] + C.C13_DELTA; js = np.where(np.abs(mz_cal - t) <= t * tol * 1e-6)[0]
                if len(js):
                    j = js[np.argmax(mean_int[js])]
                    r.update(iso_peak=A.var_names[j], iso_corr=corr_cols(img, i, [j])[0],
                             iso_ratio_obs_over_theo=(mean_int[j] / max(mean_int[i], 1e-12)) / max(b.m1_ratio, 1e-6))
        if len(cext):
            r["ext_candidates"] = "; ".join(f"{x.name}{x.adduct}({x.ppm:+.1f})" for x in cext.head(5).itertuples())
        rows.append(r)
    T = pd.DataFrame(rows)
    iso_ok = (T.iso_corr >= thr) & T.iso_ratio_obs_over_theo.between(0.5, 2.0)
    add_ok = T.adduct_corr_max >= thr
    # matrix: only matrices matched by >= 2 ions are treated as used (avoids chance matches, e.g. to 9-AA)
    mt = T.best_name[T.best_name.str.startswith("matrix:")]
    used = mt.value_counts()[lambda v: v >= 2].index.tolist()
    crep["matrix_detected"] = [u.replace("matrix:", "") for u in used]
    if len(mt) and not used:
        C.log("  no matrix identified (no matrix matched >= 2 ions)")
    unused = T.best_name.str.startswith("matrix:") & ~T.best_name.isin(used)
    T.loc[unused, "best_name"] = ""                      # chance matches to matrix ions are treated as unannotated
    is_matrix = T.best_name.isin(used) | (T.bg_ratio < 1.5)
    T["tier"] = np.select(
        [T.isotope_of != "", is_matrix, (T.best_name != "") & (add_ok | iso_ok), T.best_name != "", T.ext_candidates != ""],
        ["isotope", "matrix/background", "A", "B", "B-db"], default="C")
    # inorganic salt clusters (Na/K/Cl/phosphate): the mass defect of small organic molecules lies in a narrow range; unannotated peaks with
    # fractional m/z > 0.00115 m/z + 0.05 and (1 - fraction) > 0.05 + 0.0002 m/z cannot be organic metabolites (conservative limits)
    fr = T.mz_cal - np.floor(T.mz_cal)
    inorg = (T.best_name == "") & (fr > 0.00115 * T.mz_cal + 0.05) & ((1 - fr) > 0.05 + 0.0002 * T.mz_cal)
    T.loc[inorg & T.tier.isin(["C", "B-db"]), "tier"] = "inorganic/salt"
    salt = C.inorganic_cluster_ions(pol, mz.max() + 5); T["salt_cluster"] = ""
    for i, mzv in enumerate(T.mz_cal.values):
        d = np.abs(salt.mz.values - mzv) / mzv * 1e6; j = int(np.argmin(d))
        if d[j] <= tol and T.tier.iloc[i] in ("C", "B-db", "inorganic/salt"):
            T.loc[i, "salt_cluster"] = salt.name.values[j]; T.loc[i, "tier"] = "inorganic/salt"
    T["evidence"] = np.where(add_ok, "adducts", "") + np.where(iso_ok, "+13C", "")
    T["noise_like"] = T.morans_i < 0.05                         # essentially no spatial structure
    T["use_downstream"] = T.tier.isin(["A", "B", "B-db", "C"]) & ~T.noise_like & (T.frac_pixels >= 0.01)
    T.to_csv(os.path.join(P["TAB"], f"{name}_peak_annotation.tsv"), sep="\t", index=False)
    C.save_json(crep, os.path.join(P["TAB"], f"{name}_mass_calibration.json"))
    anc.to_csv(os.path.join(P["TAB"], f"{name}_calibration_anchors.tsv"), sep="\t", index=False)

    # ---- target metabolites
    trows = []
    for tg in TARGETS:
        th = ions[ions["name"] == tg]
        if th.empty:
            trows.append(dict(dataset=name, metabolite=tg, verdict="out of mass range / not applicable in this polarity")); continue
        hit = m[(m["name"] == tg)]
        if hit.empty:
            near = []
            for x in th.itertuples():                      # report the nearest peak for manual checking
                d = C.ppm(mz_cal, x.mz); k = int(np.argmin(np.abs(d))); near.append(f"{x.adduct}{x.mz:.4f}: nearest peak {mz_cal[k]:.4f} ({d[k]:+.0f} ppm)")
            trows.append(dict(dataset=name, metabolite=tg, verdict="not detected (no match in the peak list)", detail=" | ".join(near))); continue
        sub = T.loc[hit.peak_idx.unique()].copy()
        sub = sub[sub.best_name == tg] if (sub.best_name == tg).any() else sub
        best = sub.sort_values(["tier", "frac_pixels"], ascending=[True, False]).iloc[0]
        v = {"A": "detectable (tier A, spatially corroborated)", "B": "putative (tier B, accurate mass only)"}.get(best.tier, f"matched, but {best.tier}")
        if best.ambiguous:
            v += "; isomeric / isobaric candidates"
        if best.frac_pixels < 0.05:
            v += "; detected in < 5% of pixels"
        trows.append(dict(dataset=name, metabolite=tg, verdict=v, peak=best.peak, mz=round(best.mz_cal, 4), adduct=best.best_adduct,
                          ppm=round(best.best_ppm, 2), tier=best.tier, frac_pixels=round(best.frac_pixels, 3),
                          morans_i=round(best.morans_i, 3), adduct_partners=best.adduct_partners, candidates=best.candidates))
    TT = pd.DataFrame(trows); TT.to_csv(os.path.join(P["TAB"], f"{name}_targets.tsv"), sep="\t", index=False)

    # ---- write annotation back to the h5ad var
    for c in ("mz_cal", "best_name", "best_adduct", "best_ppm", "tier", "frac_pixels", "morans_i", "use_downstream", "isotope_of"):
        A.var[c] = T[c].values
    A.uns["mass_calibration"] = {k: v for k, v in crep.items() if k != "residual_ppm"}
    A.write_h5ad(path, compression="gzip")

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    if len(anc):
        ax[0].scatter(anc.peak_mz, anc.ppm, s=14, label="before calibration", alpha=.7)
        if "ppm_after" in anc:
            ax[0].scatter(anc.peak_mz, anc.ppm_after, s=14, label="after calibration", alpha=.7)
        ax[0].axhspan(-tol, tol, color="grey", alpha=.15); ax[0].legend(fontsize=7)
    ax[0].set_xlabel("m/z"); ax[0].set_ylabel("ppm error"); ax[0].set_title(f"{name}: anchor mass error (tol ±{tol:.1f})", fontsize=8)
    vc = T.tier.value_counts().reindex(["A", "B", "B-db", "C", "isotope", "matrix/background", "inorganic/salt"]).fillna(0)
    ax[1].bar(vc.index, vc.values, color=["#2a7", "#7b4", "#bb5", "#999", "#58a", "#c55", "#a7a"]); ax[1].set_title("annotation tiers", fontsize=8)
    ax[1].tick_params(axis="x", labelsize=7, rotation=30)
    ax[2].scatter(T.frac_pixels, T.morans_i, s=6, c=T.tier.map({"A": "#2a7", "B": "#7b4", "C": "#999"}).fillna("#c55"))
    ax[2].set_xscale("log"); ax[2].set_xlabel("fraction of tissue pixels detected"); ax[2].set_ylabel("Moran's I (binned)")
    fig.tight_layout(); fig.savefig(os.path.join(P["FIG"], f"{name}_annotation_qc.png"), dpi=150); plt.close(fig)
    C.log(f"  matrix: {crep['matrix_detected']}; tiers: " + ", ".join(f"{k} {int(v)}" for k, v in vc.items()))
    for r in TT.itertuples():
        C.log(f"    {r.metabolite:<32} {r.verdict}")
    return TT


files = [f for f in a.files.split(",") if f] or sorted(
    glob.glob(os.path.join(P["PROC"], "sbiad1426_maldi_pixels.h5ad")) + glob.glob(os.path.join(P["PROC"], "ravi_maldi", "*.h5ad")))
if not files:
    raise SystemExit("no MSI h5ad found; run p2_m1_02 / p2_m1_03 first")
allt = []
for f in files:
    try:
        allt.append(annotate(f))
    except Exception as e:
        import traceback; traceback.print_exc(); C.log(f"  failed {f}: {e}")
if allt:
    pd.concat(allt).to_csv(os.path.join(P["TAB"], "all_targets_detectability.tsv"), sep="\t", index=False)
    C.log(f"→ {P['TAB']}/all_targets_detectability.tsv")
