#!/usr/bin/env python3
"""wcol_seed.py <camp> <shapes.json>: training instances of the weighted vertex colouring problem (inside a job).

Every instance is a GENERATED graph of one of the kinds the public hard set is made of - never a public instance:
  gnp     n, p            random graph, every edge with probability p
  queen   rows, cols      squares of a board; two squares are adjacent when a queen moves between them (only board
                          sizes that are NOT in the public collection)
  planted n, k, p         k colour classes of equal size, an edge between two classes with probability p (a graph
                          with a hidden k-colouring)
  geo     n, d            n random points in the unit square, an edge when two points are closer than the radius
                          that gives the mean degree d (sparse, clustered)
Weights: integers drawn uniformly from 1..20 (the range of the public set). Then the AUTHORS' OWN REDUCTION
(upstream/wcol_dl/reduction.py, both rules, iterated - as for their reduced_wvcp files) is applied, which also puts
the vertices in decreasing weight order. Output:
  camp/<camp>/priv/train/inst/T01.wcol ..   initial.json   recipe.json (evaluation side)
OVERLAP CHECK against the 188 public instances: identical files, identical signatures (vertices, edges, weights,
degrees), same (vertices, edges). Prints aggregates only."""
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("SE_ROOT") or (Path.home() / "epicco"))
sys.path.insert(0, str(ROOT / "fw"))
from problems import wcol as P  # noqa: E402

CAMP, SHAPES = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
spec = importlib.util.spec_from_file_location("red", str(ROOT / "upstream" / "wcol_dl" / "reduction.py"))
red = importlib.util.module_from_spec(spec)
sys.modules["red"] = red                                     # its dataclasses look their module up by name
spec.loader.exec_module(red)
PUBLIC_QUEENS = {(5, 5), (6, 6), (7, 7), (8, 8), (8, 12), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16)}


def gen(sh, rng):
    k = sh["kind"]
    if k == "gnp":
        n, p = sh["n"], sh["p"]
        return n, [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    if k == "queen":
        r, c = sh["rows"], sh["cols"]
        assert (r, c) not in PUBLIC_QUEENS and (c, r) not in PUBLIC_QUEENS, "a board of the public collection"
        cells = [(i, j) for i in range(r) for j in range(c)]
        e = []
        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                (i1, j1), (i2, j2) = cells[a], cells[b]
                if i1 == i2 or j1 == j2 or abs(i1 - i2) == abs(j1 - j2):
                    e.append((a, b))
        return r * c, e
    if k == "planted":
        n, kk, p = sh["n"], sh["k"], sh["p"]
        cls = [i % kk for i in range(n)]
        rng.shuffle(cls)
        return n, [(i, j) for i in range(n) for j in range(i + 1, n) if cls[i] != cls[j] and rng.random() < p]
    if k == "geo":
        n, d = sh["n"], sh["d"]
        rad2 = d / (math.pi * (n - 1))
        pts = [(rng.random(), rng.random()) for _ in range(n)]
        return n, [(i, j) for i in range(n) for j in range(i + 1, n)
                   if (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2 < rad2]
    raise ValueError(k)


tdir = ROOT / "camp" / CAMP / "priv" / "train" / "inst"
assert not (ROOT / "camp" / CAMP).exists(), "camp/%s exists already" % CAMP
tdir.mkdir(parents=True)
work = Path(tempfile.mkdtemp())
(work / "original_graphs").mkdir()
(work / "reduced").mkdir()
os.chdir(work)
names, rec = [], []
for idx, sh in enumerate(SHAPES, 1):
    name = "T%02d" % idx
    rng = random.Random(int(sh["seed"]))
    n, edges = gen(sh, rng)
    w = [rng.randint(1, 20) for _ in range(n)]
    with open("original_graphs/%s.col" % name, "w") as f:
        f.write("p edge %d %d\n" % (n, len(edges)) + "".join("e %d %d\n" % (a + 1, b + 1) for a, b in edges))
    with open("original_graphs/%s.col.w" % name, "w") as f:
        f.write("".join("%d\n" % x for x in w))
    so = sys.stdout
    sys.stdout = open(os.devnull, "w")                      # the reduction prints one line per instance
    try:
        red.reduce_instance(name, "wvcp", True, True, True, "reduced")
    finally:
        sys.stdout.close()
        sys.stdout = so
    rn = rm = 0
    el = []
    for line in open("reduced/%s.col" % name):
        t = line.split()
        if t and t[0] == "p":
            rn, rm = int(t[2]), int(t[3])
        elif t and t[0] == "e":
            el.append("e %s %s\n" % (t[1], t[2]))
    rw = [int(x) for x in open("reduced/%s.col.w" % name).read().split()]
    assert len(rw) == rn and len(el) == rm
    (tdir / (name + ".wcol")).write_text("p edge %d %d\n" % (rn, rm) + "".join("n %d %d\n" % (i + 1, rw[i]) for i in range(rn)) + "".join(el))
    ok, why = P.validate_instance(str(tdir / (name + ".wcol")), 1, 10 ** 9)
    assert ok, (name, why)
    names.append(name)
    rec.append({"name": name, "shape": sh, "vertices": n, "edges": len(edges), "vertices_reduced": rn, "edges_reduced": rm,
                "density_reduced": 2.0 * rm / (rn * (rn - 1.0)) if rn > 1 else 0.0})
(tdir.parent / "initial.json").write_text(json.dumps(names))
(tdir.parent / "recipe.json").write_text(json.dumps({"weights": "uniform 1..20", "reduction": "the authors' reduction.py, both rules, iterated",
                                                     "instances": rec}, indent=1))
os.chdir(str(ROOT))
shutil.rmtree(str(work))
kinds = {}
for r in rec:
    kinds[r["shape"]["kind"]] = kinds.get(r["shape"]["kind"], 0) + 1
print("training instances %d: kinds %s | vertices after the reduction %s | removed by the reduction (per instance) %s | density %.3f..%.3f" % (
    len(rec), kinds, sorted(r["vertices_reduced"] for r in rec), sorted(r["vertices"] - r["vertices_reduced"] for r in rec),
    min(r["density_reduced"] for r in rec), max(r["density_reduced"] for r in rec)))
# ---- overlap with the public instances -------------------------------------------------------------------------------
pub = sorted((ROOT / "data" / "wcol" / "test").glob("inst_*/*.wcol"))
ts = [P.signature(str(tdir / (n + ".wcol"))) for n in names]
tm = {hashlib.md5((tdir / (n + ".wcol")).read_bytes()).hexdigest() for n in names}
same_file = same_sig = same_dims = 0
for f in pub:
    if hashlib.md5(f.read_bytes()).hexdigest() in tm:
        same_file += 1
    s = P.signature(str(f))
    same_sig += sum(1 for a in ts if a == s)
    same_dims += sum(1 for a in ts if a[:2] == s[:2])
print("OVERLAP CHECK training vs the %d public instances: identical files %d | identical signatures %d | same (vertices, edges) %d" % (
    len(pub), same_file, same_sig, same_dims))
print("WCOL_SEED_DONE")
