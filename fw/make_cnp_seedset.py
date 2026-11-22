#!/usr/bin/env python3
"""make_cnp_seedset.py <shapes.json> [seed] [--camp NAME] [--out DIR]: training set of the critical node island group.

The instances are NEW graphs generated here; no file, vertex or edge of any public instance is used. What is taken
from the public collection is its DESCRIPTION (four random-graph models; networks from practice: power, transport,
communication, social, biological) and GROUP AGGREGATES of its structure (ranges of the number of vertices, the mean
degree, the share K / n, the clustering).

shapes.json = [{"model": ..., "n": ..., "ks": K / n, <model parameters>}, ...]; models:
  ba      preferential attachment, `m` edges per new vertex (m = 1: a tree)
  er      uniform random graph with mean degree `md`; isolated vertices are dropped
  ws      ring lattice with `k` neighbours, every edge rewired with probability `p`
  ff      forest fire: every new vertex links to a random one and "burns" on from it with forward probability `pf`
  geo     points in the plane, a spanning tree of short edges plus the next-shortest edges up to mean degree `md`
          (power-grid-like)
  grid    a `w` x n/w lattice with a share `cut` of the edges removed, largest component (road-like)
  social  preferential attachment with triad closing (`m` edges per new vertex, closing probability `pt`)
  comm    overlapping dense groups of `gmin`..`gmax` vertices (inside probability `pin`) plus `xr` random links per vertex
  router  a preferential-attachment tree plus `x` extra links per vertex to neighbours of neighbours
  hub     `h` hubs linked to each other, every other vertex attached to 1..`a` hubs chosen by rank, plus a few
          links among the others (airline-like)
Files: <out>/inst/T00-<k>-n<n>.cnp (the solver's native format + a last line "K <k>"); names carry no model label.
initial.json = the names; recipe.json = seed, shapes and the structural figures of every instance."""
import bisect
import json
import random
import sys
from pathlib import Path

SE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import cnp as P   # noqa: E402

argv = sys.argv[1:]


def opt(name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


camp = opt("--camp", "cnp_seed")
out = Path(opt("--out", str(SE / "camp" / camp / "priv" / "train")))
shapes = json.loads(Path(argv[0]).read_text())
seed = int(argv[1]) if len(argv) > 1 else 20261119


class G:
    def __init__(self, n):
        self.n = n
        self.adj = [set() for _ in range(n)]

    def add(self, a, b):
        if a != b and b not in self.adj[a]:
            self.adj[a].add(b)
            self.adj[b].add(a)
            return True
        return False

    def m(self):
        return sum(len(a) for a in self.adj) // 2


def ba(S, rng):
    n, m = S["n"], int(S.get("m", 1))
    g = G(n)
    rep = []
    for v in range(m + 1):                              # a small clique (m = 1: one edge) to start from
        for w in range(v):
            g.add(v, w)
            rep += [v, w]
    for v in range(m + 1, n):
        tg = set()
        while len(tg) < m:
            tg.add(rep[rng.randrange(len(rep))])
        for w in tg:
            g.add(v, w)
            rep += [v, w]
    return g


def er(S, rng):
    n = S["n"]
    g = G(n)
    need = int(round(S["md"] * n / 2.0))
    while g.m() < need:
        g.add(rng.randrange(n), rng.randrange(n))
    return g


def ws(S, rng):
    n, k, p = S["n"], int(S["k"]), float(S["p"])
    g = G(n)
    for v in range(n):
        for j in range(1, k // 2 + 1):
            g.add(v, (v + j) % n)
    for v in range(n):
        for j in range(1, k // 2 + 1):
            w = (v + j) % n
            if rng.random() < p and w in g.adj[v]:
                for _ in range(50):
                    x = rng.randrange(n)
                    if x != v and x not in g.adj[v]:
                        g.adj[v].discard(w)
                        g.adj[w].discard(v)
                        g.add(v, x)
                        break
    return g


def ff(S, rng):
    n, pf = S["n"], float(S["pf"])
    g = G(n)
    g.add(0, 1)
    for v in range(2, n):
        w = rng.randrange(v)
        seen, front = {w}, [w]
        while front and len(seen) < int(S.get("burn_max", 60)):
            nxt = []
            for x in front:
                k = 0
                while rng.random() < pf:                 # geometric number of neighbours to burn on to
                    k += 1
                cand = [y for y in g.adj[x] if y not in seen]
                rng.shuffle(cand)
                for y in cand[:k]:
                    seen.add(y)
                    nxt.append(y)
            front = nxt
        for x in seen:
            g.add(v, x)
    return g


def geo(S, rng):
    n = S["n"]
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    side = max(1, int((n / 3.0) ** 0.5))
    cell = {}
    for i, (x, y) in enumerate(pts):
        cell.setdefault((int(x * side), int(y * side)), []).append(i)
    cand = []
    for i, (x, y) in enumerate(pts):
        cx, cy = int(x * side), int(y * side)
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell.get((cx + dx, cy + dy), ()):
                    if j != i:
                        near.append(((pts[j][0] - x) ** 2 + (pts[j][1] - y) ** 2, j))
        near.sort()
        for d, j in near[:int(S.get("knn", 6))]:
            cand.append((d, min(i, j), max(i, j)))
    cand = sorted(set(cand))
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    g = G(n)
    rest = []
    for d, a, b in cand:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
            g.add(a, b)
        else:
            rest.append((d, a, b))
    need = int(round(S["md"] * n / 2.0))
    noise = float(S.get("noise", 0.5))                   # not strictly the shortest: a share is skipped at random
    for d, a, b in rest:
        if g.m() >= need:
            break
        if rng.random() > noise:
            g.add(a, b)
    return g


def grid(S, rng):
    w = int(S["w"])
    h = max(2, S["n"] // w)
    n = w * h
    g = G(n)
    cut = float(S["cut"])
    for y in range(h):
        for x in range(w):
            v = y * w + x
            if x + 1 < w and rng.random() >= cut:
                g.add(v, v + 1)
            if y + 1 < h and rng.random() >= cut:
                g.add(v, v + w)
            if x + 1 < w and y + 1 < h and rng.random() < float(S.get("diag", 0.0)):
                g.add(v, v + w + 1)
    return g


def social(S, rng):
    n, m, pt = S["n"], int(S["m"]), float(S["pt"])
    g = G(n)
    rep = []
    for v in range(m + 1):
        for w in range(v):
            g.add(v, w)
            rep += [v, w]
    for v in range(m + 1, n):
        last, done = None, 0
        tries = 0
        while done < m and tries < 20 * m:
            tries += 1
            if last is not None and rng.random() < pt and g.adj[last]:
                w = rng.choice(tuple(g.adj[last]))       # close a triangle with the vertex linked to last
            else:
                w = rep[rng.randrange(len(rep))]
            if g.add(v, w):
                rep += [v, w]
                last = w
                done += 1
    return g


def comm(S, rng):
    n = S["n"]
    g = G(n)
    order = list(range(n))
    rng.shuffle(order)
    i = 0
    groups = []
    while i < n:
        s = rng.randint(int(S["gmin"]), int(S["gmax"]))
        grp = order[i:i + s]
        if groups and rng.random() < float(S.get("overlap", 0.5)):      # a few members of an earlier group join
            grp = grp + rng.sample(rng.choice(groups), min(3, s))
        groups.append(grp)
        i += s
    pin = float(S["pin"])
    for grp in groups:
        for a in range(len(grp)):
            for b in range(a):
                if rng.random() < pin:
                    g.add(grp[a], grp[b])
    for _ in range(int(float(S.get("xr", 0.3)) * n)):
        g.add(rng.randrange(n), rng.randrange(n))
    return g


def router(S, rng):
    n = S["n"]
    g = ba({"n": n, "m": 1}, rng)
    extra = int(float(S["x"]) * n)
    tries = 0
    while extra > 0 and tries < 50 * n:
        tries += 1
        v = rng.randrange(n)
        a = rng.choice(tuple(g.adj[v]))
        b = rng.choice(tuple(g.adj[a]))
        if g.add(v, b):
            extra -= 1
    return g


def hub(S, rng):
    n, h, amax = S["n"], int(S["h"]), int(S.get("a", 3))
    g = G(n)
    for a in range(h):
        for b in range(a):
            if rng.random() < float(S.get("ph", 0.6)):
                g.add(a, b)
    cum, tot = [], 0.0
    for r in range(h):
        tot += 1.0 / (r + 1) ** float(S.get("zipf", 1.0))
        cum.append(tot)
    for v in range(h, n):
        for _ in range(rng.randint(1, amax)):
            g.add(v, bisect.bisect_left(cum, rng.random() * tot))
    for _ in range(int(float(S.get("x", 0.2)) * n)):
        g.add(rng.randrange(h, n), rng.randrange(h, n))
    return g


MODELS = {"ba": ba, "er": er, "ws": ws, "ff": ff, "geo": geo, "grid": grid, "social": social, "comm": comm, "router": router, "hub": hub}


def finish(g, keep_largest):
    """Drop isolated vertices (or keep the largest component only) and renumber 0..n-1 in a random-free, stable way."""
    n = g.n
    if keep_largest:
        seen, best = [0] * n, []
        for s in range(n):
            if seen[s]:
                continue
            seen[s] = 1
            st, comp = [s], []
            while st:
                u = st.pop()
                comp.append(u)
                for w in g.adj[u]:
                    if not seen[w]:
                        seen[w] = 1
                        st.append(w)
            if len(comp) > len(best):
                best = comp
        keep = sorted(best)
    else:
        keep = [v for v in range(n) if g.adj[v]]
    idx = {v: i for i, v in enumerate(keep)}
    return [sorted(idx[w] for w in g.adj[v] if w in idx) for v in keep]


def figures(adj, K):
    n = len(adj)
    deg = [len(a) for a in adj]
    m = sum(deg) // 2
    S = [set(a) for a in adj]
    closed = paths = 0
    rng = random.Random(1)
    for v in rng.sample(range(n), min(n, 1500)):
        a = adj[v][:50]
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                paths += 1
                closed += a[j] in S[a[i]]
    seen, nc, giant = bytearray(n), 0, 0
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = 1
        st, c = [s], 0
        while st:
            u = st.pop()
            c += 1
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = 1
                    st.append(w)
        nc += 1
        giant = max(giant, c)
    return {"n": n, "m": m, "K": K, "mean_degree": round(2.0 * m / n, 3), "k_share": round(K / float(n), 4), "max_degree": max(deg),
            "degree_le_1": round(sum(d <= 1 for d in deg) / float(n), 3), "giant": round(giant / float(n), 3), "components": nc,
            "clustering": round(closed / paths, 4) if paths else 0.0}


(out / "inst").mkdir(parents=True, exist_ok=True)
names, stats = [], []
for k, S in enumerate(shapes):
    rng = random.Random(seed * 1000003 + k)
    g = MODELS[S["model"]](S, rng)
    adj = finish(g, S["model"] in ("grid", "geo") or bool(S.get("largest")))
    n = len(adj)
    K = max(1, int(round(float(S["ks"]) * n)))
    name = "T00-%02d-n%d" % (k + 1, n)
    f = out / "inst" / (name + ".cnp")
    with open(f, "w", newline="\n") as fh:
        fh.write("%d\n" % n)
        for v in range(n):
            fh.write("%d: %s\n" % (v, " ".join(str(w) for w in adj[v])))
        fh.write("K %d\n" % K)
    ok, why = P.validate_instance(str(f), 50, 10 ** 6)
    if not ok:
        sys.exit("generated instance %d is not valid: %s" % (k + 1, why))
    fg = figures(adj, K)
    names.append(name)
    stats.append(fg)
    print("%2d  n %6d  m %7d  K %5d (%.3f)  mean degree %6.2f  max degree %5d  degree<=1 %.2f  giant %.2f  clustering %.3f" % (
        k + 1, fg["n"], fg["m"], K, fg["k_share"], fg["mean_degree"], fg["max_degree"], fg["degree_le_1"], fg["giant"], fg["clustering"]))
(out / "initial.json").write_text(json.dumps(names))
(out / "recipe.json").write_text(json.dumps({"seed": seed, "generator": "make_cnp_seedset.py", "shapes": shapes, "figures": stats}))
print("written: %d instances to %s (seed %d)" % (len(names), out, seed))
