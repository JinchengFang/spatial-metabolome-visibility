import os, re
import numpy as np, pandas as pd


def gem_topology(yml, gtsv, cache=None):
    if cache and os.path.exists(cache):
        return pd.read_csv(cache, sep="\t")
    sec, r, rx = None, None, []; in_m = in_sub = False
    for line in open(yml):
        if line.startswith("- reactions:"):
            sec = "r"; continue
        if line.startswith("- genes:") or line.startswith("- compartments:") or line.startswith("- metabolites:"):
            if r is not None:
                rx.append(r); r = None
            sec = "g" if not line.startswith("- metabolites:") else "m"; continue
        if sec != "r":
            continue
        if line.startswith("  - !!omap"):
            if r is not None:
                rx.append(r)
            r = dict(mets=[], rule="", sub=[]); in_m = in_sub = False; continue
        if r is None:
            continue
        if re.match(r"\s{4}- metabolites:", line):
            in_m, in_sub = True, False; continue
        if re.match(r"\s{4}- subsystem:", line):
            in_m, in_sub = False, True; continue
        if re.match(r"\s{4}- \S", line):
            in_m = in_sub = False
            m = re.match(r"\s{4}- gene_reaction_rule: (.*)", line)
            if m:
                r["rule"] = m.group(1)
            continue
        if in_m:
            m = re.match(r"\s{6}- (\S+): ", line)
            if m:
                r["mets"].append(m.group(1))
        elif in_sub:
            m = re.match(r"\s{6}- (.*)", line)
            if m:
                r["sub"].append(m.group(1).strip())
    if r is not None:
        rx.append(r)
    cnt = {}
    for r in rx:
        has_gene = bool(re.search(r"ENSG\d+", r["rule"]))
        comps = {}
        for sid in r["mets"]:
            comps.setdefault(sid[:-1], set()).add(sid[-1])
        tr_sub = any(("transport" in s.lower()) or ("exchange" in s.lower()) for s in r["sub"])
        for base, cs in comps.items():
            c = cnt.setdefault(base, dict(n_chem_gene=0, n_export_gene=0, n_internal_transport_gene=0, n_total=0, comps=set()))
            c["n_total"] += 1; c["comps"] |= cs
            if not has_gene:
                continue
            if len(cs) >= 2 and "e" in cs:
                c["n_export_gene"] += 1                       # trans-membrane: the metabolite occurs in e and in an intracellular compartment
            elif len(cs) >= 2 or tr_sub:
                c["n_internal_transport_gene"] += 1           # between organelles
            else:
                c["n_chem_gene"] += 1                         # reactions that change the chemical structure
    T = pd.DataFrame([dict(met_id=k, **{kk: (len(vv) if kk == "comps" else vv) for kk, vv in v.items()}) for k, v in cnt.items()])
    T = T.rename(columns={"comps": "n_compartments"})
    T["export_share"] = T.n_export_gene / np.maximum(T.n_export_gene + T.n_chem_gene, 1)
    if cache:
        T.to_csv(cache, sep="\t", index=False)
    return T
