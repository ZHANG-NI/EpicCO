"""overlap_check.py <problem> <training dir> <test dir> [<test dir> ...]: is any training instance a test instance?
(aggregates only - no instance is named). Three checks per training instance against EVERY test instance:
  1. identical file bytes (md5);
  2. identical content up to relabelling: the problem plugin's signature (cvrp: size, capacity, customer locations,
     demand multiset; jssp: shape + the jobs' duration multisets; bpp2d: bin types + item types; mwis: sizes) and its
     similarity (1.0 = a copy; for cvrp also the share of customer locations taken from a test instance);
  3. identical file NAME (stem).
Run from the framework directory's parent:  python3 cluster/overlap_check.py bpp2d camp/bpp_G1/priv/train/inst data/bpp2d/test/class ..."""
import hashlib
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / (os.environ.get("SE_FW") or "framework")))
P = importlib.import_module("problems." + sys.argv[1])
tr = sorted(Path(sys.argv[2]).rglob(P.INSTANCE_GLOB))
te = []
for d in sys.argv[3:]:
    te += sorted(Path(d).rglob(P.INSTANCE_GLOB))
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()   # noqa: E731


def sig(p):
    try:
        return P.signature(str(p))
    except Exception:
        return None


T = [(md5(p), sig(p), p.stem) for p in te]
bad_sig = sum(s is None for _, s, _ in T)
tm, tn = {h for h, _, _ in T}, {n for _, _, n in T}
same_bytes = same_content = same_name = 0
best = []
for p in tr:
    s = sig(p)
    same_bytes += md5(p) in tm
    same_name += p.stem in tn
    b = 0.0
    if s is not None:
        for _, q, _ in T:
            if q is not None:
                b = max(b, P.sig_similarity(s, q))
                if b >= 1.0:
                    break
    best.append(b)
    same_content += b >= 1.0
print("%s: %d training instances (%s) against %d test instances (%d directories; %d unreadable)" % (
    sys.argv[1], len(tr), sys.argv[2], len(te), len(sys.argv) - 3, bad_sig))
print("  identical bytes: %d | identical content up to relabelling: %d | identical file name: %d | largest similarity of a training instance to any test instance: %.4f (mean %.4f)" % (
    same_bytes, same_content, same_name, max(best) if best else 0.0, (sum(best) / len(best)) if best else 0.0))
print("  RESULT: %s" % ("NO OVERLAP" if (same_bytes == 0 and same_content == 0) else "OVERLAP FOUND"))
