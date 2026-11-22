"""CVRP plugin: instance parsing, independent solution checking, build and run commands of the host solver."""
import math
import os
import re

NAME = "cvrp"
OBJECTIVE = "min"
INSTANCE_GLOB = "*.vrp"


def parse(path):
    hdr, xy, dem, depots = {}, {}, {}, []
    sec = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            u = s.upper()
            if u.startswith("NODE_COORD_SECTION"):
                sec = "xy"; continue
            if u.startswith("DEMAND_SECTION"):
                sec = "dem"; continue
            if u.startswith("DEPOT_SECTION"):
                sec = "dep"; continue
            if u.startswith("EOF"):
                break
            if u.startswith("EDGE_WEIGHT_SECTION"):
                sec = "skip"; continue
            if sec is None or (":" in s and not s[0].isdigit() and s[0] != "-"):
                if ":" in s:
                    k, v = s.split(":", 1)
                    hdr[k.strip().upper()] = v.strip()
                    sec = None
                continue
            p = s.split()
            if sec == "xy" and len(p) >= 3:
                xy[int(p[0])] = (float(p[1]), float(p[2]))
            elif sec == "dem" and len(p) >= 2:
                dem[int(p[0])] = float(p[1])
            elif sec == "dep":
                v = int(float(p[0]))
                if v > 0:
                    depots.append(v)
    dim = int(hdr["DIMENSION"])
    ewt = hdr.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if ewt != "EUC_2D":
        raise ValueError("unsupported EDGE_WEIGHT_TYPE %s" % ewt)
    if len(xy) != dim or len(dem) != dim:
        raise ValueError("coordinate/demand count does not match DIMENSION")
    depot = depots[0] if depots else 1
    if depot != 1:
        raise ValueError("depot must be node 1")
    dur = hdr.get("DISTANCE") or hdr.get("DURATION")
    return {"n": dim - 1, "cap": float(hdr["CAPACITY"]), "xy": xy, "dem": dem, "duration": dur, "hdr": hdr}


def size(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.upper().startswith("DIMENSION"):
                return int(line.split(":")[1]) - 1
    raise ValueError("no DIMENSION")


def time_limit(n):
    return 0.4 * n


def verify(inst_path, sol_path, opts=None):
    """Recompute the cost of a solution file (`Route #k: ...` lines). Returns {"ok", "cost", "msg", "routes"}."""
    opts = opts or {}
    rounded = bool(opts.get("round", 1))
    try:
        I = parse(inst_path)
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "instance: %r" % (ex,)}
    if not os.path.exists(sol_path):
        return {"ok": False, "cost": None, "msg": "no solution file"}
    routes = []
    with open(sol_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"\s*Route\s*#?\s*\d+\s*:\s*(.*)$", line, flags=re.I)
            if m:
                try:
                    r = [int(x) for x in m.group(1).split()]
                except ValueError:
                    return {"ok": False, "cost": None, "msg": "bad route line"}
                if r:
                    routes.append(r)
    if not routes:
        return {"ok": False, "cost": None, "msg": "no routes"}
    n, cap, xy, dem = I["n"], I["cap"], I["xy"], I["dem"]
    seen = set()
    total = 0.0
    for r in routes:
        load = 0.0
        prev = 1
        for c in r:
            if c < 1 or c > n:
                return {"ok": False, "cost": None, "msg": "customer %d out of range" % c}
            if c in seen:
                return {"ok": False, "cost": None, "msg": "customer %d visited twice" % c}
            seen.add(c)
            node = c + 1
            load += dem[node]
            d = math.hypot(xy[prev][0] - xy[node][0], xy[prev][1] - xy[node][1])
            total += math.floor(d + 0.5) if rounded else d
            prev = node
        d = math.hypot(xy[prev][0] - xy[1][0], xy[prev][1] - xy[1][1])
        total += math.floor(d + 0.5) if rounded else d
        if load > cap + 1e-9:
            return {"ok": False, "cost": None, "msg": "capacity exceeded (%g > %g)" % (load, cap)}
    if len(seen) != n:
        return {"ok": False, "cost": None, "msg": "%d customers missing" % (n - len(seen))}
    k = opts.get("veh")
    if k and len(routes) > int(k):
        return {"ok": False, "cost": None, "msg": "%d routes > fleet %d" % (len(routes), int(k))}
    cost = int(total) if rounded else round(total, 2)
    return {"ok": True, "cost": cost, "msg": "", "routes": len(routes)}


def validate_instance(path, lo, hi):
    """Is a GENERATED instance a well-formed member of the benchmark's problem class?"""
    try:
        I = parse(path)
    except Exception as ex:
        return False, "parse: %r" % (ex,)
    n = I["n"]
    if n < lo or n > hi:
        return False, "n=%d outside [%d, %d]" % (n, lo, hi)
    if I["duration"]:
        return False, "duration constraint not part of the problem class"
    if I["dem"][1] != 0:
        return False, "depot demand must be 0"
    cap = I["cap"]
    if cap <= 0 or cap != int(cap):
        return False, "capacity must be a positive integer"
    pts = set()
    for i in range(1, n + 2):
        x, y = I["xy"][i]
        if x != int(x) or y != int(y) or x < 0 or y < 0 or x > 1000000 or y > 1000000:
            return False, "coordinates must be non-negative integers"
        pts.add((x, y))
        if i > 1:
            d = I["dem"][i]
            if d != int(d) or d < 1 or d > cap:
                return False, "customer demand must be an integer in [1, capacity]"
    if len(pts) < max(3, (n + 1) // 2):
        return False, "too many coincident points"
    return True, ""


def features(path):
    I = parse(path)
    n = I["n"]
    tot = sum(I["dem"][i] for i in range(2, n + 2))
    return {"n": n, "min_routes": int(math.ceil(tot / I["cap"])), "cust_per_route": round(n / max(1.0, tot / I["cap"]), 2),
            "coord_max": max(max(x, y) for x, y in I["xy"].values()),
            "demand_max_over_cap": round(max(I["dem"][i] for i in range(2, n + 2)) / I["cap"], 4)}


# features of a generated instance that must stay inside the public benchmark's range
RANGE_FEATURES = ["cust_per_route", "demand_max_over_cap"]
RANGE_UPPER = ["coord_max"]


def canonical(path_in, path_out, rng=None):
    """Re-write an instance in one fixed layout with no name/comment. With `rng`: also apply a random symmetry of the
    coordinate square and a random renumbering of the customers (the instance is the same problem, but it can no
    longer be recognised as a particular file)."""
    I = parse(path_in)
    n = I["n"]
    pts = [I["xy"][i] for i in range(1, n + 2)]
    dem = [I["dem"][i] for i in range(1, n + 2)]
    if rng is not None:
        M = 1000.0 if max(max(p) for p in pts) <= 1000 else float(max(max(p) for p in pts))
        k = rng.randrange(8)
        def tr(p):
            x, y = p
            if k & 1:
                x = M - x
            if k & 2:
                y = M - y
            if k & 4:
                x, y = y, x
            return (x, y)
        pts = [tr(p) for p in pts]
        order = list(range(1, n + 1))
        rng.shuffle(order)
        pts = [pts[0]] + [pts[i] for i in order]
        dem = [dem[0]] + [dem[i] for i in order]
    f = lambda v: str(int(v)) if v == int(v) else repr(v)     # noqa: E731
    # three header lines before DIMENSION: the host solver's reader skips exactly three lines
    L = ["NAME : instance", "COMMENT : none", "TYPE : CVRP", "DIMENSION : %d" % (n + 1), "EDGE_WEIGHT_TYPE : EUC_2D",
         "CAPACITY : %s" % f(I["cap"]), "NODE_COORD_SECTION"]
    L += ["%d %s %s" % (i + 1, f(p[0]), f(p[1])) for i, p in enumerate(pts)]
    L.append("DEMAND_SECTION")
    L += ["%d %s" % (i + 1, f(d)) for i, d in enumerate(dem)]
    L += ["DEPOT_SECTION", "1", "-1", "EOF"]
    with open(path_out, "w") as fo:
        fo.write("\n".join(L) + "\n")


def signature(path):
    I = parse(path)
    n = I["n"]
    return (n, I["cap"], frozenset(I["xy"][i] for i in range(2, n + 2)), tuple(sorted(I["dem"][i] for i in range(2, n + 2))))


def sig_similarity(a, b):
    """How much of instance a is taken from instance b (1.0 = copy): share of a's customer locations that are also
    customer locations of b, or 1.0 when size, capacity and the multiset of demands are all identical."""
    if a[0] == b[0] and a[1] == b[1] and a[3] == b[3]:
        return 1.0
    return len(a[2] & b[2]) / float(max(1, len(a[2])))


BUILD = r'''set -e
g++ -O3 -DNDEBUG -std=gnu++17 -IProgram Program/*.cpp -o "$1/solver"
'''


def run_cmd(bindir, inst_path, sol_path, T, seed, opts=None):
    opts = opts or {}
    cmd = [os.path.join(bindir, "solver"), inst_path, sol_path, "-t", "%.2f" % T, "-seed", str(seed),
           "-round", str(int(opts.get("round", 1)))]
    if opts.get("veh"):
        cmd += ["-veh", str(int(opts["veh"]))]
    return cmd


def trace(sol_path, stdout_path):
    """Anytime curve of one run: [(seconds, cost)] from the solver's search-progress export (<sol>.PG.csv)."""
    out = []
    p = sol_path + ".PG.csv"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            q = line.strip().split(";")
            if len(q) >= 4:
                try:
                    out.append((float(q[3]), float(q[2])))
                except ValueError:
                    pass
    return out


if __name__ == "__main__":
    import json
    import sys
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "verify":
        print(json.dumps(verify(a[1], a[2], {"round": 1})))
    elif len(a) >= 2 and a[0] == "validate":
        print(validate_instance(a[1], int(a[2]) if len(a) > 2 else 1, int(a[3]) if len(a) > 3 else 10**6))
    elif len(a) >= 2 and a[0] == "features":
        print(json.dumps(features(a[1])))
    else:
        print("usage: problem.py verify <instance> <solution> | validate <instance> [min_n max_n] | features <instance>")
