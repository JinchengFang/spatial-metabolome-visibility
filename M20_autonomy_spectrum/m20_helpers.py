import re
import numpy as np, pandas as pd

SYN = {"norepinephrine": ["noradrenaline"], "DOPAC": ["3,4-dihydroxyphenylacetate"], "homovanillic acid": ["homovanillate"], "5-HIAA": ["5-hydroxyindoleacetate"],
       "glutamate": ["l-glutamate", "glutamate"], "glutamine": ["l-glutamine", "glutamine"], "tyrosine": ["l-tyrosine", "tyrosine"], "alanine": ["l-alanine", "alanine"],
       "3-methoxytyramine": ["3-methoxytyramine"], "acetylcholine": ["acetylcholine"]}
RELEASED = {"dopamine", "norepinephrine", "serotonin", "GABA", "glutamate", "glycine", "acetylcholine", "histamine", "adenosine", "N-acetylaspartylglutamate", "ATP"}
AMINO = {"alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamate", "glutamine", "glycine", "histidine", "leucine/isoleucine", "lysine",
         "methionine", "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine", "valine"}


def name_variants(nm):
    """Common name -> candidate lower-case names (stereo prefixes removed, 'X-ic acid' -> 'X-ate', '(spectral match)' removed)."""
    n = re.sub(r"\s*\(spectral ?match\)\s*", "", str(nm)).strip().lower(); out = [n]
    base = re.sub(r"^(dl-|d-|l-|d \(\+\) |l \(\+\) |d \(-\) )", "", n); out.append(base)
    for x in (n, base):
        if x.endswith(" acid"):
            s = x[:-5]
            if s.endswith("ic"):
                out += [s[:-2] + "ate", "l-" + s[:-2] + "ate"]
            out.append(s)
        if x.endswith("acid") and not x.endswith(" acid"):          # "citricacid"
            s = x[:-4]
            if s.endswith("ic"):
                out.append(s[:-2] + "ate")
    return list(dict.fromkeys(out + ["l-" + base]))


def make_mapper(GEM, C, TOPO, max_rxn=400):
    """Returns gem_ids(name) and topo_of(ids); TOPO is indexed by met_id (m20_gem_topology.tsv)."""
    def gem_ids(nm):
        nm0 = re.sub(r"\s*\[[^\]]*\][+-]?\s*$", "", str(nm).strip())
        cands = SYN.get(nm0, []) + C.GEM_SYNONYM.get(nm0, []) + name_variants(nm0)
        ids = set()
        for c in cands:
            ids |= GEM["name2ids"].get(c.lower(), set())
        return {i for i in ids if GEM["n_rxn"].get(i, 0) <= max_rxn}

    def topo_of(ids):
        t = TOPO.reindex(sorted(ids)).dropna(how="all")
        if not len(t):
            return dict(n_chem_gene=np.nan, n_export_gene=np.nan, n_internal_gene=np.nan, n_total=np.nan, export_share=np.nan)
        ch, ex, it = float(t.n_chem_gene.sum()), float(t.n_export_gene.sum()), float(t.n_internal_transport_gene.sum())
        return dict(n_chem_gene=ch, n_export_gene=ex, n_internal_gene=it, n_total=float(t.n_total.sum()), export_share=ex / max(ch + ex, 1))
    return gem_ids, topo_of


ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])


def batch_X(Gz, sets, k):
    out = []
    for c0 in range(0, len(sets), 50):
        A = np.stack([Gz[:, s_] for s_ in sets[c0:c0 + 50]])
        if A.shape[2] > k:
            U, S_, _ = np.linalg.svd(A, full_matrices=False); A = U[:, :, :k] * S_[:, None, :k]
        out.append(A)
    return np.concatenate(out)


def batch_cv_r2(Xb, y, folds):
    N, n, p = Xb.shape; pred = np.empty((N, n)); I_ = np.eye(p)
    for f_ in np.unique(folds):
        te = folds == f_; tr = ~te
        Xt = Xb[:, tr]; mu = Xt.mean(1, keepdims=True); sd = Xt.std(1, keepdims=True) + 1e-9
        Xt = (Xt - mu) / sd; Xs = (Xb[:, te] - mu) / sd; yt = y[tr] - y[tr].mean()
        XtX = np.matmul(Xt.transpose(0, 2, 1), Xt); Xty = np.matmul(Xt.transpose(0, 2, 1), yt[None, :, None])[:, :, 0]
        best = np.full(N, np.inf); bb = np.zeros((N, p))
        for al in ALPHAS:
            Minv = np.linalg.inv(XtX + al * I_); b = np.matmul(Minv, Xty[:, :, None])[:, :, 0]
            fit = np.matmul(Xt, b[:, :, None])[:, :, 0]; h = (np.matmul(Xt, Minv) * Xt).sum(2)
            loo = (((yt[None, :] - fit) / (1 - h)) ** 2).mean(1); m = loo < best; best[m] = loo[m]; bb[m] = b[m]
        pred[:, te] = np.matmul(Xs, bb[:, :, None])[:, :, 0] + y[tr].mean()
    return 1 - ((y[None, :] - pred) ** 2).sum(1) / ((y - y.mean()) ** 2).sum()


def matched_sets(own, uni, dec, n, rng):
    uni = np.setdiff1d(uni, own); by = {d: uni[dec[uni] == d] for d in np.unique(dec[uni])}; need = pd.Series(dec[own]).value_counts(); out = []
    for _ in range(n):
        s = []
        for d, k in need.items():
            pool = by.get(d, np.array([], int)); pool = pool if len(pool) >= k else uni
            s.append(rng.choice(pool, k, replace=False))
        out.append(np.concatenate(s))
    return out
