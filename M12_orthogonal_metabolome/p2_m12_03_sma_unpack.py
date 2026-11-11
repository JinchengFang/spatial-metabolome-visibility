#!/usr/bin/env python3

import argparse, glob, os, re, sys, tarfile, zipfile
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--sma-dir", default=None)
a = ap.parse_args(); P = C.paths(a.root); TAB, FIG = C.step_dirs(P, "Step9")
D = a.sma_dir or os.path.join(P["PUB"], "Vicari2024_SMA_same_section")
if not os.path.isdir(D):
    C.log(f"{D} not found; run p2_step9_downloads.sh first"); sys.exit(0)
UNP = os.path.join(D, "_unpacked"); os.makedirs(UNP, exist_ok=True)
for f in sorted(glob.glob(os.path.join(D, "*", "*"))):
    base = os.path.basename(f); tgt = os.path.join(UNP, re.sub(r"\.(zip|tar\.gz|tgz|tar)$", "", base))
    if os.path.exists(tgt):
        continue
    try:
        if f.endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                z.extractall(tgt)
        elif f.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(f) as t:
                t.extractall(tgt, filter="data") if sys.version_info >= (3, 12) else t.extractall(tgt)
        else:
            continue
        C.log(f"unpacked {base}")
    except Exception as e:
        C.log(f"could not unpack {base}: {e}")
allf = [p for p in glob.glob(os.path.join(D, "**", "*"), recursive=True) if os.path.isfile(p) and not p.endswith(".part")]
inv = pd.DataFrame(dict(path=[os.path.relpath(p, D) for p in allf], size_mb=[round(os.path.getsize(p) / 1e6, 2) for p in allf]))
inv["ext"] = inv.path.str.extract(r"(\.[A-Za-z0-9]+(?:\.gz)?)$")[0].fillna("")
inv.to_csv(os.path.join(TAB, "sma_inventory.tsv"), sep="\t", index=False)
C.log(f"{len(inv)} files, {inv.size_mb.sum() / 1e3:.1f} GB -> {TAB}/sma_inventory.tsv")
