#!/usr/bin/env python3

import argparse, os, re, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C, p2_loaders as L
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None); ap.add_argument("--datasets", default="IDHmut,ccRCC,GBM"); ap.add_argument("--res", type=int, default=200)
ap.add_argument("--ccrcc-dir", default=None); ap.add_argument("--ion-set", default="strict"); ap.add_argument("--include-uncertain", action="store_true")
ap.add_argument("--max-feat", type=int, default=150); ap.add_argument("--min-det", type=float, default=0.10); ap.add_argument("--cluster-r", type=float, default=0.9)
ap.add_argument("--module-r", type=float, default=0.3, help="module cut: mean residual correlation >= this value")
ap.add_argument("--min-module", type=int, default=5); ap.add_argument("--n-perm", type=int, default=499); ap.add_argument("--block-um", type=float, default=1000.0)
ap.add_argument("--min-px", type=int, default=2); ap.add_argument("--no-shift", action="store_true"); ap.add_argument("--min-samples", type=int, default=3)
ap.add_argument("--step", default="Step9", help="output folder (same-section data: --step Step10 --datasets SMA)")
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, a.step); R = a.res
RDIR = os.path.join(P["PROC"], "m12_residuals"); os.makedirs(RDIR, exist_ok=True)
cfg = L.Config(P, ion_set=a.ion_set, ccrcc_dir=a.ccrcc_dir, include_uncertain=a.include_uncertain)
DS = [d.strip() for d in a.datasets.split(",") if d.strip()]; t0 = time.time(); plt.rcParams.update({"font.size": 7}); rng = np.random.default_rng(0)

# m/z differences between chemically homologous ions (isotopes / adducts / water loss / polarity pairs of the same compound)
SAME_POL = [1.00336, 2.00671, 21.98194, 37.95588, 15.97394, 18.01056, 17.02655, 35.97668, 43.98983, 46.00548, 27.99491, 33.98772]
OPP_POL = [2.01456, 23.99650, 39.97044, 33.96157, 11.98007, 3.99431, 15.99600]


def chem_related(mz, pol):
    mz = np.asarray(mz, float); n = len(mz); M = np.zeros((n, n), bool)
    if not np.isfinite(mz).all():
        return M
    D = np.abs(mz[:, None] - mz[None, :]); tol = 0.003 + 5e-6 * np.maximum(mz[:, None], mz[None, :])
    same = (np.asarray(pol)[:, None] == np.asarray(pol)[None, :])
    for d in SAME_POL:
        M |= same & (np.abs(D - d) < tol)
    for d in OPP_POL:
        M |= ~same & (np.abs(D - d) < tol)
    np.fill_diagonal(M, True)
    return M


def eigengene(M):
    U, S_, Vt = np.linalg.svd(C.zcols(M), full_matrices=False)
    return U[:, 0] * S_[0] * np.sign(Vt[0].sum() or 1.0)


def raster_r2(v, xy, R):
    """Acquisition-raster control: R² of an eigengene on bin-row + bin-column fixed effects (additive two-way ANOVA); stripes / row-column drift -> close to 1."""
    rr = np.round(xy[:, 1] / R).astype(int); cc_ = np.round(xy[:, 0] / R).astype(int)
    X = np.column_stack([np.ones(len(v))] + [(rr == k).astype(float) for k in np.unique(rr)[1:]] + [(cc_ == k).astype(float) for k in np.unique(cc_)[1:]])
    fit = X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return float(1 - ((v - fit) ** 2).sum() / ((v - v.mean()) ** 2).sum())


def adduct_of(ann):
    m = re.search(r"\[(M[^\]]*)\][+-]", str(ann).split(";")[0])
    return m.group(1) if m else ""


def lipid_class(ann):
    first = str(ann).split(";")[0].strip()
    m = re.match(r"^(FA|CAR|LPC|LPE|LPI|LPG|LPS|PC|PE|PS|PI|PA|PG|SM|Cer|HexCer|ST|CE|TG|DG|MG|SHexCer|NAE|PE-Cer)\s", first)
    if m:
        return m.group(1)
    return "small molecule" if first else "unannotated"


# ------------------------------------------------------------------ shifts (Step 8)
SHIFT = {}
fs = os.path.join(P["P2"], "05_tables", "Step8", "m11_registration_summary.tsv")
if os.path.exists(fs) and not a.no_shift:
    rs = pd.read_csv(fs, sep="\t"); rs = rs[(rs.res_um == rs.res_um.min()) & (rs.cv_gain > 0.02)]
    SHIFT = {r.sample: (float(r.cv_shift_dx), float(r.cv_shift_dy)) for r in rs.itertuples()}
    C.log(f"cross-fitted Step 8 shifts applied (gain > 0.02): {SHIFT if SHIFT else 'none'}")

# ------------------------------------------------------------------ residuals per sample
store, frows = {}, []
for S in L.load_all(cfg, DS):
    G = L.make_grid(S, R, cfg)
    if G is None:
        continue
    sh = SHIFT.get(S["sample"], (0.0, 0.0))
    Y, c0 = L.pixel_means(S, G, None, sh); YA, cA = L.pixel_means(S, G, S["px_half"] == 0, sh); YB, cB = L.pixel_means(S, G, S["px_half"] == 1, sh)
    ok = (c0 >= a.min_px) & (cA >= a.min_px) & (cB >= a.min_px)
    kk = np.floor((S["px_xy"] + np.asarray(sh) - G["o"]) / R).astype(np.int64); ipx = G["gi"].reindex(kk[:, 1] * 100000 + kk[:, 0]).fillna(-1).astype(int).values
    v_ = ipx >= 0; tic_bin = np.log1p(np.bincount(ipx[v_], weights=S["px_tic"][v_], minlength=len(G["keys"])) / np.maximum(np.bincount(ipx[v_], minlength=len(G["keys"])), 1))[ok]
    Bk, LG, src = L.tx_blocks(S, G["CB"], G["ncell"], G["comp_fixed"])
    X = np.column_stack([np.asarray(v, float) for v in Bk.values()])[ok]; X = X[:, X.std(0) > 1e-9]
    Y, YA, YB, xy = Y[ok], YA[ok], YB[ok], G["xy"][ok]
    det = (np.expm1(Y) > 0).mean(0) if S["feat_group"] is None else np.ones(Y.shape[1])
    okf = np.where(det >= a.min_det)[0]
    names = np.asarray(S["names"]).astype(str)[okf]
    mz = np.asarray(S["mz"], float)[okf] if S["mz"] is not None else np.full(len(okf), np.nan)
    pol = (np.asarray(S["pol"], dtype=object)[okf] if not isinstance(S["pol"], str) else np.array([S["pol"]] * len(okf), dtype=object)).astype(str)
    ann = names if S["annotated"] else np.array([C.annotate_mz(m_, p_, cfg.LM) for m_, p_ in zip(mz, pol)], dtype=object).astype(str)
    Z, ZA, ZB = C.zcols(Y[:, okf]), C.zcols(YA[:, okf]), C.zcols(YB[:, okf])
    folds = C.spatial_folds(xy, a.block_um, seed=R)
    E = Z - C.cv_ridge_gcv(X, Z, folds); EA = ZA - C.cv_ridge_gcv(X, ZA, folds); EB = ZB - C.cv_ridge_gcv(X, ZB, folds)
    r2 = 1 - (E ** 2).sum(0) / (Z ** 2).sum(0)
    _, SBy = C.split_half_reliability(ZA, ZB); rE, SBe = C.split_half_reliability(EA, EB)
    npm = 999 if Z.shape[1] > 100 else 99
    IY, pY = C.moran_knn(Z, xy, k=6, n_perm=npm, seed=1); IE, pE = C.moran_knn(E, xy, k=6, n_perm=npm, seed=2)
    qY, qE = C.bh(pY), C.bh(pE)
    structured = qY < 0.05
    for j in range(len(okf)):
        frows.append(dict(dataset=S["dataset"], sample=S["sample"], res_um=R, feature=names[j], mz=mz[j], polarity=pol[j], annotation=ann[j],
                          lipid_class=lipid_class(ann[j]), adduct=adduct_of(ann[j]), structured=bool(structured[j]), R2_tx=r2[j],
                          reliability_SB=SBy[j], residual_r_AB=rE[j], residual_SB=SBe[j], moran_Y=IY[j], moran_resid=IE[j], q_moran_resid=qE[j],
                          orthogonal_reliable=bool(structured[j] and np.nan_to_num(SBe[j]) >= 0.5 and qE[j] < 0.05)))
    st = np.where(structured)[0]
    C.log(f"  {S['dataset']} {S['sample']} @ {R} µm{' (shift ' + str(sh) + ' µm)' if any(sh) else ''}: {len(xy)} bins, {len(st)} structured features; "
          f"median R²_tx {np.median(r2[st]):+.3f}; median residual reliability SB {np.nanmedian(SBe[st]):.2f}; median residual Moran {np.median(IE[st]):.2f} (raw {np.median(IY[st]):.2f}); "
          f"reliable and structured orthogonal variation: {np.mean([frows[-len(okf) + j]['orthogonal_reliable'] for j in st]):.0%}")
    store[S["sample"]] = dict(dataset=S["dataset"], names=names, mz=mz, pol=pol, ann=ann, Z=Z, E=E, xy=xy, IY=IY, qY=qY, tic=tic_bin)
    np.savez_compressed(os.path.join(RDIR, f"{S['dataset']}_{S['sample']}_{R}um.npz"), names=names, mz=mz, pol=pol, ann=ann.astype(str), Z=Z.astype(np.float32),
                        E=E.astype(np.float32), xy=xy, moran_Y=IY, q_moran_Y=qY, tic=tic_bin, dataset=S["dataset"], sample=S["sample"], res=R)
FR = pd.DataFrame(frows)
if not len(FR):
    raise SystemExit("no results")
FR.to_csv(os.path.join(TAB, "m12_residual_features.tsv"), sep="\t", index=False)
SUM = FR[FR.structured].groupby(["dataset", "sample"]).agg(n_structured=("feature", "size"), R2_tx=("R2_tx", "median"), reliability_SB=("reliability_SB", "median"),
                                                           residual_SB=("residual_SB", "median"), moran_Y=("moran_Y", "median"), moran_resid=("moran_resid", "median"),
                                                           frac_orthogonal_reliable=("orthogonal_reliable", "mean")).reset_index()
SUM["D37_orthogonal_structured"] = SUM.frac_orthogonal_reliable >= 0.5

# ------------------------------------------------------------------ ccRCC: conservation across patients
cc = sorted(s for s, v in store.items() if v["dataset"] == "ccRCC")
pair_rows, lopo_rows, mod_rows, mem_rows, maps = [], [], [], [], {}
if len(cc) >= a.min_samples:
    common = sorted(set.intersection(*[set(store[s]["names"]) for s in cc]))
    ix = {s: pd.Index(store[s]["names"]).get_indexer(common) for s in cc}
    nstr = np.sum([store[s]["qY"][ix[s]] < 0.05 for s in cc], axis=0)
    cand = np.where(nstr >= max(2, int(np.ceil(0.6 * len(cc)))))[0]
    Zp = np.vstack([store[s]["Z"][:, ix[s][cand]] for s in cc]); mI = np.median([store[s]["IY"][ix[s][cand]] for s in cc], axis=0)
    lab = fcluster(linkage(squareform(1 - np.abs(np.nan_to_num(np.corrcoef(Zp.T))), checks=False), "average"), t=1 - a.cluster_r, criterion="distance")
    reps = sorted([np.where(lab == k)[0][np.argmax(mI[lab == k])] for k in np.unique(lab)], key=lambda j: -mI[j])[: a.max_feat]
    F = cand[reps]; feat = [common[j] for j in F]; s0 = cc[0]
    mzF = store[s0]["mz"][ix[s0][F]]; polF = store[s0]["pol"][ix[s0][F]]; annF = store[s0]["ann"][ix[s0][F]]
    CHEM = chem_related(mzF, polF); iu = np.triu_indices(len(F), 1); keep = ~CHEM[iu]
    Craw = {s: np.nan_to_num(np.corrcoef(store[s]["Z"][:, ix[s][F]].T)) for s in cc}
    Cres = {s: np.nan_to_num(np.corrcoef(store[s]["E"][:, ix[s][F]].T)) for s in cc}
    C.log(f"[ccRCC conservation] {len(cc)} patients, {len(common)} shared features -> {len(cand)} structured in >= {int(np.ceil(0.6 * len(cc)))} -> {len(F)} after merging; "
          f"{int((~keep).sum())}/{len(keep)} chemically homologous pairs excluded")
    for i_, s1 in enumerate(cc):
        for s2 in cc[i_ + 1:]:
            row = dict(sample_1=s1, sample_2=s2)
            for kind, Cm in (("raw", Craw), ("residual", Cres)):
                v1 = Cm[s1][iu][keep]; v2 = Cm[s2][iu][keep]; rho = spearmanr(v1, v2).correlation
                null = []
                for _ in range(a.n_perm):
                    p_ = rng.permutation(len(F)); M2 = Cm[s2][np.ix_(p_, p_)]
                    null.append(spearmanr(v1, M2[iu][keep]).correlation)
                row[f"rho_{kind}"] = rho; row[f"p_{kind}"] = (np.sum(np.array(null) >= rho) + 1) / (a.n_perm + 1)
            pair_rows.append(row)
    PR = pd.DataFrame(pair_rows)
    C.log("  residual covariation agreement (homologous pairs removed): median rho " + f"{PR.rho_residual.median():.2f} (raw {PR.rho_raw.median():.2f}); patient pairs with p < 0.05 {int((PR.p_residual < 0.05).sum())}/{len(PR)}")

    def modules(Cs_list):
        Zm = np.mean([np.arctanh(np.clip(Cm, -.999, .999)) for Cm in Cs_list], 0); Rm = np.tanh(Zm); np.fill_diagonal(Rm, 1)
        lab_ = fcluster(linkage(squareform(np.clip(1 - Rm, 0, 2), checks=False), "average"), t=1 - a.module_r, criterion="distance")
        return [np.where(lab_ == k)[0] for k in np.unique(lab_) if (lab_ == k).sum() >= a.min_module], Rm

    def coherence(Cm, mem):
        sub = Cm[np.ix_(mem, mem)]; return sub[np.triu_indices(len(mem), 1)].mean()

    def adduct_driven(mem):
        ad_ = [adduct_of(annF[j]) for j in mem if adduct_of(annF[j])]
        if len(ad_) < 3:
            return False, ""
        top = pd.Series(ad_).value_counts()
        return bool(top.iloc[0] / len(ad_) > 0.6 and top.index[0] not in ("M+H", "M-H")), top.index[0]

    RB = float(np.median([np.median([raster_r2(store[s]["Z"][:, ix[s][j]], store[s]["xy"], R) for j in F[:40]]) for s in cc]))   # raster R² baseline of raw metabolite features
    # leave one patient out: modules discovered on four patients, tested in the fifth
    for h in cc:
        mods, _ = modules([Cres[s] for s in cc if s != h])
        for mi, mem in enumerate(mods):
            obs = coherence(Cres[h], mem)
            null = np.array([coherence(Cres[h], rng.choice(len(F), len(mem), replace=False)) for _ in range(a.n_perm)])
            drv, top = adduct_driven(mem)
            lopo_rows.append(dict(held_out=h, module=mi, size=len(mem), coherence_heldout=obs, null_mean=null.mean(), p=(np.sum(null >= obs) + 1) / (a.n_perm + 1),
                                  adduct_driven=drv, top_adduct=top, members=";".join(feat[j] for j in mem[:20])))
    LO = pd.DataFrame(lopo_rows)
    # final modules (five patients) and per-patient agreement; a module is covered by the LOPO validation if Jaccard >= 0.5 with a replicated LOPO module
    mods, Rm = modules([Cres[s] for s in cc])
    for mi, mem in enumerate(mods):
        fs_ = set(feat[j] for j in mem); rep = 0
        for h in cc:
            g = LO[(LO.held_out == h) & (LO.p < 0.05)]
            if any(len(fs_ & set(m.split(";"))) / max(1, len(fs_ | set(m.split(";")))) >= 0.3 for m in g.members):
                rep += 1
        drv, top = adduct_driven(mem); cls = pd.Series([lipid_class(annF[j]) for j in mem]).value_counts()
        coh = {s: coherence(Cres[s], mem) for s in cc}; cohr = {s: coherence(Craw[s], mem) for s in cc}
        # technical control: correlation of the module eigengene with the total ion current of the bin (ionization / suppression); median |r| > 0.5 flagged
        tic_r = float(np.median([abs(np.corrcoef(eigengene(store[s]["E"][:, ix[s][F[mem]]]), store[s]["tic"])[0, 1]) for s in cc]))
        rast = float(np.median([raster_r2(eigengene(store[s]["E"][:, ix[s][F[mem]]]), store[s]["xy"], R) for s in cc]))
        mod_rows.append(dict(module=f"OM{mi + 1}", size=len(mem), n_patients_replicated_LOPO=rep, adduct_driven=drv, top_adduct=top, tic_abs_r=tic_r, tic_linked=tic_r > 0.5,
                             raster_R2=rast, raster_R2_baseline=RB, raster_like=bool(rast >= 0.6 and rast > RB + 0.2),
                             polarity=";".join(f"{k}:{v}" for k, v in pd.Series(polF[mem]).value_counts().items()),
                             classes="; ".join(f"{k} {v}" for k, v in cls.items()), mz_range=f"{np.nanmin(mzF[mem]):.1f}–{np.nanmax(mzF[mem]):.1f}",
                             mean_residual_coherence=float(np.mean(list(coh.values()))), mean_raw_coherence=float(np.mean(list(cohr.values()))),
                             **{f"coh_{s}": coh[s] for s in cc}))
        for j in mem:
            mem_rows.append(dict(module=f"OM{mi + 1}", feature=feat[j], mz=mzF[j], polarity=polF[j], annotation=annF[j], lipid_class=lipid_class(annF[j]),
                                 adduct=adduct_of(annF[j]), mean_corr_in_module=float(np.mean(Rm[j, mem]))))
        maps[f"OM{mi + 1}"] = {s: (store[s]["xy"], eigengene(store[s]["E"][:, ix[s][F[mem]]])) for s in cc}
    MO = pd.DataFrame(mod_rows)
    if len(MO):
        MO["conserved"] = (MO.n_patients_replicated_LOPO >= int(np.ceil(0.8 * len(cc)))) & ~MO.adduct_driven & ~MO.tic_linked & ~MO.raster_like
        MO = MO.sort_values(["conserved", "n_patients_replicated_LOPO", "mean_residual_coherence"], ascending=False)
    for df, f in ((PR, "m12_conservation_pairs.tsv"), (LO, "m12_module_lopo.tsv"), (MO, "m12_modules.tsv"), (pd.DataFrame(mem_rows), "m12_module_members.tsv")):
        df.to_csv(os.path.join(TAB, f), sep="\t", index=False)
    if len(MO):
        C.log("orthogonal modules:\n" + MO[["module", "size", "n_patients_replicated_LOPO", "adduct_driven", "top_adduct", "tic_abs_r", "raster_R2", "raster_like", "classes", "mz_range",
                                   "mean_residual_coherence", "conserved"]].round(3).to_string(index=False))
    SUM["D38_conserved_pairs"] = f"{int((PR.p_residual < 0.05).sum())}/{len(PR)}"
    SUM["D38_orthogonal_covariation_conserved"] = bool((PR.p_residual < 0.05).mean() >= 0.7)
    SUM["D39_conserved_modules"] = int(MO.conserved.sum()) if len(MO) else 0
    SUM["D39_conserved_orthogonal_module"] = bool(len(MO) and MO.conserved.any())
else:
    C.log(f"fewer than {a.min_samples} ccRCC samples; conservation across patients skipped")
SUM.to_csv(os.path.join(TAB, "m12_orthogonal_summary.tsv"), sep="\t", index=False)
C.log("summary:\n" + SUM.round(3).to_string(index=False))

# ------------------------------------------------------------------ figures
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1, 1.3]))
st = FR[FR.structured]; keys = st[["dataset", "sample"]].drop_duplicates().values.tolist()
for k_, (ds, smp) in enumerate(keys):
    e = st[(st["sample"] == smp)]
    axs[0].boxplot([e.reliability_SB.dropna(), e.residual_SB.dropna()], positions=[k_ * 3, k_ * 3 + 1], widths=.7, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#d9d9d9"), medianprops=dict(color="k"))
axs[0].set_xticks([k_ * 3 + .5 for k_ in range(len(keys))]); axs[0].set_xticklabels([f"{d}\n{s}" for d, s in keys], fontsize=5.5)
axs[0].set_ylabel("split-half reliability (left: metabolite, right: transcriptome residual)"); axs[0].axhline(.5, c="k", ls="--", lw=.6)
axs[0].set_title("a  transcriptome-orthogonal variance is reliable", fontsize=7)
if pair_rows:
    PR = pd.DataFrame(pair_rows); x = np.arange(len(PR))
    axs[1].scatter(PR.rho_raw, PR.rho_residual, c=np.where(PR.p_residual < 0.05, "#de2d26", "#969696"), s=18)
    lim = [min(0, PR[["rho_raw", "rho_residual"]].min().min()) - .05, 1]; axs[1].plot(lim, lim, "k--", lw=.5)
    axs[1].set_xlabel("covariation conservation (raw)"); axs[1].set_ylabel("conservation of residual covariation"); axs[1].set_title("b  patient pairs (red: p<0.05)", fontsize=7)
if mod_rows:
    MO_ = pd.DataFrame(mod_rows); H = MO_[[c for c in MO_ if c.startswith("coh_")]].values
    im = axs[2].imshow(H, aspect="auto", cmap="RdBu_r", vmin=-.6, vmax=.6); plt.colorbar(im, ax=axs[2], fraction=.04)
    axs[2].set_yticks(range(len(MO_))); axs[2].set_yticklabels([f"{r.module} (n={r.size}, rep {r.n_patients_replicated_LOPO}{', adduct' if r.adduct_driven else ''})" for r in MO_.itertuples()], fontsize=5)
    axs[2].set_xticks(range(len(cc))); axs[2].set_xticklabels(cc, fontsize=6); axs[2].set_title("c  orthogonal modules: residual coherence per patient", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "m12_orthogonal_modules.png"), dpi=170); plt.close(fig)
if maps:
    top = list(pd.DataFrame(mod_rows).sort_values("n_patients_replicated_LOPO", ascending=False).module[:3])
    fig, axs = plt.subplots(len(top), len(cc), figsize=(2.2 * len(cc), 2.2 * len(top)), squeeze=False)
    for i_, m in enumerate(top):
        for j_, s in enumerate(cc):
            xy, v = maps[m][s]; axs[i_, j_].scatter(xy[:, 0], xy[:, 1], c=np.clip(C.zcols(v[:, None])[:, 0], -2.5, 2.5), cmap="RdBu_r", s=4, marker="s")
            axs[i_, j_].set_aspect("equal"); axs[i_, j_].axis("off"); axs[i_, j_].set_title(f"{m} · {s}", fontsize=6)
    fig.suptitle("Transcriptome-orthogonal metabolic modules (residual eigengene)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "m12_module_maps.png"), dpi=160); plt.close(fig)
C.log(f"done ({time.time() - t0:.0f} s)")
