#!/usr/bin/env python3
"""make_glass_seedset.py <families.json> --camp NAME [--overlap <dir of public .gfam files> [--seeds N]]
Training set of the glass-cutting islands = FAMILIES of order books (the solver has no random number, so the seed of a
run selects the generated instance of a family; see fw/problems/glass.py). Writes camp/NAME/priv/train/inst/<name>.gfam
(one recipe file per family), initial.json (the names) and recipe.json (the recipes with structural figures over the
first 20 seeds). File work only.

families.json: a list of recipes {"name", "n": [lo, hi], and either "stacks": [lo, hi] or "chain_share" +
"chain_len": [lo, hi]; optional "dup" (share of repeated sizes, default 0.5), "scale" (item size factor, default 1)}.

--overlap: every family x seeds 1..N (default 300) and the seeds of the evaluation blocks that follow the pattern
1000 k + offset + i is too many to list, so the check is statistical by construction and exact on the sample: the
signature (items, stacks, multiset of item sizes) of each generated instance is compared with every public instance."""
import collections
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import glass as P   # noqa: E402

args = sys.argv[1:]
fams = json.loads(Path(args[0]).read_text())
camp = args[args.index("--camp") + 1]
SE = Path(os.environ.get("SE_ROOT", "."))
tdir = SE / "camp" / camp / "priv" / "train" / "inst"
tdir.mkdir(parents=True, exist_ok=True)
names, rec = [], []
for f in fams:
    fam = dict(f, kind="family")
    p = tdir / (fam["name"] + ".gfam")
    p.write_text(json.dumps(fam))
    ok, why = P.validate_instance(str(p), 1, 10 ** 6)
    if not ok:
        sys.exit("family %s: %s" % (fam["name"], why))
    n, stk, plates, dup = [], [], [], []
    for s in range(1, 21):
        b = P.gen_instance(fam, s)[0]
        ok2 = all(min(int(r.split(";")[1]), int(r.split(";")[2])) >= 100 and max(int(r.split(";")[1]), int(r.split(";")[2])) <= 3210 for r in b[1:])
        if not ok2:
            sys.exit("family %s seed %d: an item does not fit" % (fam["name"], s))
        sz = collections.Counter((max(int(r.split(";")[1]), int(r.split(";")[2])), min(int(r.split(";")[1]), int(r.split(";")[2]))) for r in b[1:])
        n.append(len(b) - 1)
        stk.append(len({r.split(";")[3] for r in b[1:]}))
        plates.append(P.item_area(b) / (6000.0 * 3210.0))
        dup.append(sum(v for v in sz.values() if v > 1) / float(len(b) - 1))
    names.append(fam["name"])
    rec.append({"family": fam, "items": [min(n), max(n)], "stacks": [min(stk), max(stk)], "plates_of_items": round(st.mean(plates), 2),
                "repeated_sizes": round(st.mean(dup), 2)})
    print("%-8s items %3d..%3d | stacks %3d..%3d | item area %.1f plates | items with a repeated size %.0f %%" % (
        fam["name"], min(n), max(n), min(stk), max(stk), st.mean(plates), 100 * st.mean(dup)))
(tdir.parent / "initial.json").write_text(json.dumps(names))
(tdir.parent / "recipe.json").write_text(json.dumps(rec, indent=1))
print("families written: %d -> %s" % (len(names), tdir))
if "--overlap" in args:
    pub = sorted(Path(args[args.index("--overlap") + 1]).rglob("*.gfam"))
    N = int(args[args.index("--seeds") + 1]) if "--seeds" in args else 300
    sig = {tuple(P.signature(str(p))[:3]) for p in pub}
    full = {tuple(P.signature(str(p))) for p in pub}
    byname = {p.stem for p in pub}
    hit = hit_full = 0
    for f in fams:
        fam = dict(f, kind="family")
        for s in range(1, N + 1):
            g = P.signature_of(P.gen_instance(fam, s)[0])
            hit += tuple(g[:3]) in sig
            hit_full += tuple(g) in full
    print("OVERLAP CHECK: %d families x seeds 1..%d = %d generated instances against %d public instances (%d distinct order books): "
          "same items + stacks + multiset of item sizes %d, same with the same stack lengths %d, same name %d" % (
              len(fams), N, len(fams) * N, len(pub), len(sig), hit, hit_full, sum(f["name"] in byname for f in fams)))
