#!/usr/bin/env python3
"""make_mwis_seedset3.py <shapes.json> [seed] [--camp NAME] [--out DIR] [--conv PATH]: training set of the maximum weight
independent set island group, third generator. (The first two gave graphs on which the original solver had almost
nothing left to gain - rejected by measurement on 11-15: in both, the value of a route was a sum over its loads and
every load had an obvious best driver.)

The instances are NEW, generated here; no file, vertex, edge or weight of any public instance is used. What is
taken from the public collection is its DESCRIPTION (routes of many solutions of a randomised long-haul scheduling
heuristic are pooled; a route = one driver and the loads assigned to it; two routes conflict when they share the
driver or a load) and GROUP AGGREGATES.

One instance: hubs in the plane; `nd` drivers with a home near a hub; `lpd` x nd loads, each with a pickup point
near one hub and a drop point near another. A route is a CHAIN: the driver goes from home to the first pickup, from
every drop to the next pickup, and home again. Weight = revenue (loaded distance of its loads, with some noise)
minus `dh` x the empty distance - so it depends on which loads are chained and in which order, not on the loads one
by one. A SCHEDULE is built by a randomised greedy: drivers in random order; each starts with one of the nearest free
pickups and keeps appending one of the free pickups nearest to its last drop (the j-th nearest free one with
probability proportional to gamma^j), up to a chain length drawn for that driver. Schedules are drawn until `n`
distinct routes exist (at most `smax` schedules). The conflict graph is the union of the cliques (one per driver,
one per load), built by our converter (cluster/mwis_conv.c), which also writes the binary files of the verifier.
shapes.json = [{"n", "nd", "lpd", "kd", "gamma", "kmax", "dh", "scale", "smax"}, ...]; file names carry no group label."""
import json
import random
import subprocess
import sys
from pathlib import Path

SE = Path(__file__).resolve().parent.parent
argv = sys.argv[1:]


def opt(name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


camp = opt("--camp", "mwis_seed3")
out = Path(opt("--out", str(SE / "camp" / camp / "priv" / "train")))
conv = opt("--conv", str(SE / "bin" / "mwis" / "mwis_conv"))
shapes = json.loads(Path(argv[0]).read_text())
seed = int(argv[1]) if len(argv) > 1 else 20261117
if out.exists():
    sys.exit("%s already exists: not overwritten" % out)
rng = random.Random(seed)


def knn(pts, qs, k):
    """For every query point the k nearest of pts (indices, nearest first), by a grid."""
    n = len(pts)
    g = max(1, int((n / 4.0) ** 0.5))
    cell = {}
    for i, (x, y) in enumerate(pts):
        cell.setdefault((min(g - 1, int(x * g)), min(g - 1, int(y * g))), []).append(i)
    res = []
    for (x, y) in qs:
        cx, cy = min(g - 1, int(x * g)), min(g - 1, int(y * g))
        r = 1
        while True:
            cand = [i for a in range(cx - r, cx + r + 1) for b in range(cy - r, cy + r + 1) for i in cell.get((a, b), ())]
            if len(cand) >= k or r > g:
                break
            r += 1
        cand.sort(key=lambda i: (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2)
        res.append(cand[:k])
    return res


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def make(name, S):
    nd, kd, kmax = int(S["nd"]), int(S["kd"]), int(S["kmax"])
    nl = int(round(nd * S["lpd"]))
    G = max(4, int(nd ** 0.5 / 2))
    hub = [(rng.random(), rng.random()) for _ in range(G)]
    sp = 0.3 / G ** 0.5
    clip = lambda v: min(0.999999, max(0.0, v))   # noqa: E731
    near = lambda c: (clip(rng.gauss(c[0], sp)), clip(rng.gauss(c[1], sp)))   # noqa: E731
    home = [near(rng.choice(hub)) for _ in range(nd)]
    pick, drop = [], []
    for _ in range(nl):
        a, b = rng.sample(range(G), 2)
        pick.append(near(hub[a]))
        drop.append(near(hub[b]))
    rev = [dist(pick[l], drop[l]) * rng.lognormvariate(0.0, 0.15) for l in range(nl)]
    first = knn(pick, home, kd)
    nxt = [[m for m in row if m != l][:kd] for l, row in enumerate(knn(pick, drop, kd + 1))]
    gamma, dh = float(S["gamma"]), float(S["dh"])
    lens = list(range(1, kmax + 1))
    lw = [1.0 / (1.0 + abs(k - S["lpd"])) ** 2 for k in lens]            # chain length drawn around loads per driver
    routes, nsol = {}, 0
    order = list(range(nd))
    while len(routes) < S["n"] and nsol < S["smax"]:
        nsol += 1
        rng.shuffle(order)
        used = bytearray(nl)
        for d in order:
            cap = rng.choices(lens, lw)[0]
            chain, cands = [], first[d]
            while len(chain) < cap:
                m = None
                for x in cands:
                    if not used[x]:
                        m = x
                        if rng.random() >= gamma:
                            break
                if m is None:
                    break
                used[m] = 1
                chain.append(m)
                cands = nxt[m]
            if chain:
                key = (d, tuple(chain))
                if key not in routes:
                    routes[key] = len(routes)
    keys = sorted(routes, key=routes.get)[:int(S["n"])]
    rng.shuffle(keys)
    n = len(keys)
    raw = []
    for d, ch in keys:
        empty = dist(home[d], pick[ch[0]]) + sum(dist(drop[a], pick[b]) for a, b in zip(ch, ch[1:])) + dist(drop[ch[-1]], home[d])
        raw.append(max(0.0, sum(rev[l] for l in ch) - dh * empty))
    med = sorted(raw)[n // 2] or 1.0
    unit = S["scale"] / med
    cap = float(S.get("wcap", 2.4))                                  # no weight above wcap x the median (very long hauls)
    W = [int(round(unit * min(r, cap * med))) for r in raw]
    dm, lm = [[] for _ in range(nd)], [[] for _ in range(nl)]
    for v, (d, ch) in enumerate(keys, 1):
        dm[d].append(v)
        for l in ch:
            lm[l].append(v)
    tmp = out / "tmp" / name
    tmp.mkdir(parents=True)
    with open(tmp / "node_weights.txt", "w") as f:
        f.write("".join("%d %d\n" % (v, w) for v, w in enumerate(W, 1)))
    cl = [c for c in dm + lm if c]
    rng.shuffle(cl)
    with open(tmp / "cliques.txt", "w") as f:
        f.write("".join(" ".join(map(str, c)) + "\n" for c in cl))
    r = subprocess.run([conv, str(tmp), str(out / "inst" / name)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("converter failed on %s: %s" % (name, r.stderr[-300:]))
    st = json.loads(r.stdout.strip().split("\n")[-1])
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    st.update(schedules=nsol, drivers=nd, loads=nl, routes_per_driver=n / float(nd), zero_weights=sum(w == 0 for w in W))
    return st


(out / "inst").mkdir(parents=True)
names, stats = [], []
for k, S in enumerate(shapes, 1):
    name = "T00-%02d" % k
    st = make(name, S)
    name2 = "%s-n%d" % (name, st["n"])
    for p in list((out / "inst").glob(name + ".graph*")):
        p.rename(p.with_name(p.name.replace(name, name2, 1)))
    names.append(name2)
    stats.append(st)
    print("%2d: n %d | m %d | mean degree %.1f (max %d) | drivers %d, loads %d, schedules %d, routes per driver %.1f | clique size median %d / p90 %d / max %d | cliques per vertex median %d, p90 %d, max %d | weight median %d, p10 %.2f, p90 %.2f, max %.2f of it, zero %d" % (
        k, st["n"], st["m"], st["degree"]["mean"], st["degree"]["max"], st["drivers"], st["loads"], st["schedules"], st["routes_per_driver"],
        st["clique_size"]["p50"], st["clique_size"]["p90"], st["clique_size"]["max"], st["cliques_per_vertex"]["p50"], st["cliques_per_vertex"]["p90"],
        st["cliques_per_vertex"]["max"], st["weight"]["p50"], st["weight"]["p10"] / max(1.0, st["weight"]["p50"]),
        st["weight"]["p90"] / max(1.0, st["weight"]["p50"]), st["weight"]["max"] / max(1.0, st["weight"]["p50"]), st["zero_weights"]), flush=True)
try:
    (out / "tmp").rmdir()
except OSError:
    pass
(out / "initial.json").write_text(json.dumps(names))
(out / "recipe.json").write_text(json.dumps({"seed": seed, "generator": 3, "shapes": shapes, "figures": stats}))
print("%s: %d training instances written to %s (seed %d)" % (camp, len(names), out, seed))
