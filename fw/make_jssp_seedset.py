#!/usr/bin/env python3
"""make_jssp_seedset.py <shapes.json> [seed]: the fixed training set of the job-shop island group, written as the
training directory of a pseudo-campaign `jssp_seed` so that setup_camp.py can copy it (train_copy_from=jssp_seed).

The instances are NEW draws from the published recipes of the two public families (they are not benchmark files):
  u  every job visits the machines in a uniformly random order, durations uniform on 1..99
  r  the same routing, durations uniform on 1..200
  c  two-stage shop: the machines are split into two halves, a job visits a random order of the first half and then a
     random order of the second half; durations uniform on 1..200
shapes.json = [[family, jobs, machines], ...]; file names carry no family label (T00-<k>-n<operations>)."""
import json
import random
import sys
from pathlib import Path

SE = Path(__file__).resolve().parent.parent
shapes = json.loads(Path(sys.argv[1]).read_text())
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261113
out = SE / "camp" / "jssp_seed" / "priv" / "train"
if out.exists():
    sys.exit("jssp_seed already exists: not overwritten")
(out / "inst").mkdir(parents=True)
rng = random.Random(seed)
names = []
for k, (fam, n, m) in enumerate(shapes, 1):
    rows = []
    for j in range(n):
        if fam in ("u", "r"):
            order = list(range(m))
            rng.shuffle(order)
        else:
            h = m // 2
            a, b = list(range(h)), list(range(h, m))
            rng.shuffle(a)
            rng.shuffle(b)
            order = a + b
        hi = 99 if fam == "u" else 200
        rows.append(" ".join("%d %d" % (mc, rng.randint(1, hi)) for mc in order))
    name = "T00-%02d-n%d" % (k, n * m)
    (out / "inst" / (name + ".txt")).write_text("%d %d\n%s\n" % (n, m, "\n".join(rows)))
    names.append(name)
(out / "initial.json").write_text(json.dumps(names))
(out / "recipe.json").write_text(json.dumps({"seed": seed, "shapes": shapes}))
print("jssp_seed: %d training instances written (seed %d)" % (len(names), seed))
