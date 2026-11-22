#!/usr/bin/env python3
"""Weighted vertex colouring: colour the vertices of a graph so that no edge joins two vertices of the same colour;
the cost of a colouring is the SUM OVER THE COLOURS OF THE LARGEST VERTEX WEIGHT in the colour. Minimise it (the
number of colours is free). The solver is an open-source C++ tree search over partial colourings that completes
each visited partial colouring with a local search (directory `code/`), single thread, used as published;
`params.txt` holds its options; `ext/` holds two small header libraries it uses.

Instance (`<name>.wcol`, plain text):
  p edge <n> <m>            n vertices 1..n, m edges
  n <vertex> <weight>       one line per vertex, a positive integer weight; the vertices are numbered by
                            decreasing weight (ties: by decreasing degree)
  e <u> <v>                 one line per edge
Solution file: n lines; line i is the colour (an integer from 0) of vertex i.
A solution is valid when every vertex has a colour and no edge has both ends in one colour; its cost is recomputed
by our own verifier. Smaller is better."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

NAME = "wcol"
OBJECTIVE = "min"
INSTANCE_GLOB = "*.wcol"
TIME_LIMIT_S = 300.0
RESERVE_S = 3.0             # of the time limit: reading and writing; the program counts whole seconds

LOCKED = {"ext/fmt/os.h": "a13810d64a6d0f92bd4072811bcf0130", "ext/fmt/std.h": "9b3cd20efd652ff0bd0188f0728ba74c"}
SIDE_EXT = ()


def parse(path):
    """(n, weights [n], edges [(u, v)] 0-based)."""
    n = None
    w, edges = {}, []
    with open(path, encoding="latin-1") as f:
        for line in f:
            t = line.split()
            if not t or t[0] == "c":
                continue
            if t[0] == "p":
                n = int(t[2])
            elif t[0] == "n":
                w[int(t[1]) - 1] = int(t[2])
            elif t[0] == "e":
                edges.append((int(t[1]) - 1, int(t[2]) - 1))
            else:
                raise ValueError("unknown line type %r" % t[0])
    if n is None or len(w) != n or any(i not in w for i in range(n)):
        raise ValueError("header or vertex weights incomplete")
    if any(not (0 <= u < n and 0 <= v < n) or u == v for u, v in edges):
        raise ValueError("an edge names a vertex that does not exist, or a loop")
    return n, [w[i] for i in range(n)], edges


def size(path):
    with open(path, encoding="latin-1") as f:
        for line in f:
            if line.startswith("p "):
                return int(line.split()[2])
    raise ValueError("no p line")


def time_limit(n):
    return TIME_LIMIT_S


def io_slack(path):
    return 10.0


def read_solution(sol_path, n):
    out = []
    with open(sol_path, encoding="latin-1") as f:
        for line in f:
            t = line.strip()
            if t == "":
                continue
            out.append(int(t))
    if len(out) != n:
        raise ValueError("the solution holds %d colours, the instance %d vertices" % (len(out), n))
    if any(c < 0 for c in out):
        raise ValueError("a vertex has no colour")
    return out


def evaluate(n, w, edges, col):
    for u, v in edges:
        if col[u] == col[v]:
            raise ValueError("vertices %d and %d are adjacent and share colour %d" % (u + 1, v + 1, col[u]))
    best = {}
    for i in range(n):
        c = col[i]
        if w[i] > best.get(c, 0):
            best[c] = w[i]
    return sum(best.values()), len(best)


def verify(inst_path, sol_path, opts=None):
    """{"ok", "cost", "msg"}: a proper colouring; cost = sum over the colours of the largest weight."""
    try:
        if not os.path.exists(sol_path) or os.path.getsize(sol_path) == 0:
            return {"ok": False, "cost": None, "msg": "no solution file"}
        n, w, edges = parse(inst_path)
        try:
            cost, k = evaluate(n, w, edges, read_solution(sol_path, n))
        except ValueError as ex:
            return {"ok": False, "cost": None, "msg": "BAD %s" % ex}
        return {"ok": True, "cost": float(cost), "msg": "cost=%d colours=%d" % (cost, k), "colours": k}
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "verifier: %r" % (ex,)}


def validate_instance(path, lo, hi):
    try:
        n, w, edges = parse(path)
    except Exception as ex:
        return False, "unreadable: %r" % (ex,)
    if not (lo <= n <= hi):
        return False, "%d vertices outside %d..%d" % (n, lo, hi)
    if any(w[i] < w[i + 1] for i in range(n - 1)):
        return False, "the vertices are not numbered by decreasing weight"
    if min(w) <= 0:
        return False, "a weight is not positive"
    return True, ""


def features(path):
    n, w, edges = parse(path)
    es = set((min(u, v), max(u, v)) for u, v in edges)
    return {"n": n, "density": 2.0 * len(es) / float(max(1, n * (n - 1))), "weights_distinct_share": len(set(w)) / float(n),
            "weight_max_over_mean": max(w) * n / float(sum(w)), "mean_degree": 2.0 * len(es) / float(n)}


def canonical(path_in, path_out, rng=None):
    import shutil
    shutil.copy(path_in, path_out)


def signature(path):
    """[vertices, edges, md5 of the sorted weights, md5 of the sorted degree sequence]: equal for two files that are the
    same graph up to a renumbering of the vertices; different signatures prove that two instances differ."""
    n, w, edges = parse(path)
    es = set((min(u, v), max(u, v)) for u, v in edges)
    deg = [0] * n
    for u, v in es:
        deg[u] += 1
        deg[v] += 1
    return [n, len(es), hashlib.md5(json.dumps(sorted(w)).encode()).hexdigest(), hashlib.md5(json.dumps(sorted(deg)).encode()).hexdigest()]


def sig_similarity(a, b):
    return 1.0 if list(a) == list(b) else 0.0


RANGE_FEATURES = ["density", "weights_distinct_share"]

FORBIDDEN_OPTIONS = ("problem", "instance", "rand_seed", "time_limit", "target", "output_directory", "help")
_OPT_RE = r"^[A-Za-z_]+\s+\S+$"

# Build: the published flags (-O3, link-time optimisation, C++17); every source file of code/, four at a time.
BUILD = r'''set -e
set -o pipefail
CXX=${CXX:-g++}
command -v "$CXX" > /dev/null || { echo "no C++ compiler ($CXX) on PATH"; exit 1; }
for f in code/main.cpp code/methods/MCTS.cpp ext/cxxopts.hpp ext/fmt/core.h params.txt; do
  [ -f "$f" ] || { echo "$f is missing"; exit 1; }
done
python3 - <<'PY' || exit 1
import re, sys
seen = set()
for l in open("params.txt").read().splitlines():
    l = l.strip()
    if not l or l.startswith("#"):
        continue
    if not re.match(r"__OPT_RE__", l):
        sys.exit("params.txt: a line must read  option value  (found: %s)" % l)
    k = l.split()[0]
    if k in __FORBIDDEN__:
        sys.exit("params.txt: %s is set by the launcher and may not appear here" % k)
    if k in seen:
        sys.exit("params.txt: %s occurs twice" % k)
    seen.add(k)
PY
INC="-Iext -Icode -DFMT_HEADER_ONLY"
if [ "$2" = check ]; then                   # compile errors only, no binary
  for f in code/main.cpp code/*/*.cpp; do
    "$CXX" -O0 -DNDEBUG -std=c++17 -w -fsyntax-only $INC "$f" 2>&1 | tail -n 60
  done
  exit 0
fi
O="$1/obj"; rm -rf "$O"; mkdir -p "$O"
FLAGS="-O3 -flto=auto -DNDEBUG -std=c++17 -w $INC"
export CXX FLAGS O
ls code/main.cpp code/*/*.cpp | xargs -P 4 -I{} sh -c '"$CXX" $FLAGS -c "{}" -o "$O/$(echo "{}" | tr / _).o" 2>&1 | tail -n 60; test -s "$O/$(echo "{}" | tr / _).o"'
"$CXX" -O3 -flto=auto -s "$O"/*.o -o "$1/prog" 2>&1 | tail -n 60
rm -rf "$O"
cp params.txt "$1/"
cat > "$1/solver" <<'LAUNCHER'
#!/usr/bin/env python3
"""solver <instance.wcol> <solution> -t <seconds> -seed <int>: one run of the program (one thread). The program reads
an instance by NAME from ../instances/reduced_wvcp/<name>.col (edges) and <name>.col.w (weights, one per line)
relative to its working directory, and writes a CSV file whose last line holds its best colouring; this launcher
writes those two files into <solution>.work/, runs the program there, turns the last CSV line into the solution
file (one colour per vertex) and removes the directory. The target option is always 0 (never reached)."""
import os
import shutil
import signal
import subprocess
import sys
here = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
inst, sol = os.path.abspath(a[0]), os.path.abspath(a[1])
T = float(a[a.index("-t") + 1])
seed = a[a.index("-seed") + 1]
W = sol + ".work"
shutil.rmtree(W, ignore_errors=True)
for d in ("build", "instances/reduced_wvcp", "out/tbt"):
    os.makedirs(os.path.join(W, d))
name = "g"
n = 0
wts, edges = {}, []
with open(inst) as f:
    for line in f:
        t = line.split()
        if not t:
            continue
        if t[0] == "p":
            n = int(t[2])
        elif t[0] == "n":
            wts[int(t[1])] = t[2]
        elif t[0] == "e":
            edges.append("e %s %s\n" % (t[1], t[2]))
with open(os.path.join(W, "instances", "reduced_wvcp", name + ".col"), "w") as f:
    f.write("p edge %d %d\n" % (n, len(edges)))
    f.writelines(edges)
with open(os.path.join(W, "instances", "reduced_wvcp", name + ".col.w"), "w") as f:
    f.write("".join("%s\n" % wts[i] for i in range(1, n + 1)))
cmd = [os.path.join(here, "prog"), "--problem", "wvcp", "--instance", name, "--rand_seed", seed,
       "--time_limit", "%d" % max(1, int(T - __RESERVE__)), "--target", "0", "--output_directory", os.path.join(W, "out")]
for l in open(os.path.join(here, "params.txt")).read().splitlines():
    l = l.strip()
    if l and not l.startswith("#"):
        k, v = l.split()
        cmd += ["--" + k, v]
p = subprocess.Popen(cmd, cwd=os.path.join(W, "build"))


def stop(signum, frame):
    try:
        p.terminate()                 # the program then ends its current step and writes its last line
    except OSError:
        pass


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
rc = p.wait()
last = None
for fn in ("%s_%s.csv" % (name, seed), "%s_%s.csv.running" % (name, seed)):
    q = os.path.join(W, "out", fn)
    if os.path.exists(q):
        with open(q) as f:
            for line in f:
                if line.startswith("#") or line.startswith("turn,") or not line.strip():
                    continue
                last = line.strip()
                t = last.split(",")
                if len(t) >= 4:
                    print("TRACE %s %s" % (t[1], t[-2]))
        break
if last:
    cols = last.split(",")[-1].split(":")
    if len(cols) == n:
        with open(sol, "w") as f:
            f.write("".join("%s\n" % c for c in cols))
shutil.rmtree(W, ignore_errors=True)
sys.exit(rc if rc >= 0 else 1)
LAUNCHER
chmod +x "$1/solver"
'''.replace("__OPT_RE__", _OPT_RE).replace("__FORBIDDEN__", repr(FORBIDDEN_OPTIONS)).replace("__RESERVE__", repr(RESERVE_S))


def run_cmd(bindir, inst_path, sol_path, T, seed, opts=None):
    return [os.path.join(bindir, "solver"), inst_path, sol_path, "-t", "%d" % max(1, int(T)), "-seed", str(seed)]


def trace(sol_path, stdout_path):
    """[[seconds, cost]]: the launcher prints `TRACE <seconds> <cost>` for every improving line of the program."""
    out = []
    try:
        with open(stdout_path, encoding="latin-1", errors="ignore") as f:
            for line in f:
                if line.startswith("TRACE "):
                    q = line.split()
                    try:
                        out.append([float(q[1]), float(q[2])])
                    except (ValueError, IndexError):
                        pass
    except OSError:
        pass
    return out


def locked_violation(tree):
    """None, or why the tree is refused: a locked library file differs from the original or is missing."""
    tree = Path(tree)
    for rel, md5 in LOCKED.items():
        p = tree / rel
        if not p.exists():
            return "BUILD REFUSED: %s is missing; it must stay exactly as in the original." % rel
        if hashlib.md5(p.read_bytes()).hexdigest() != md5:
            return ("BUILD REFUSED: %s differs from the original. This library file must stay byte-identical; put your changes "
                    "in the files of code/." % rel)
    return None


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "verify":
        r = verify(a[1], a[2])
        print(json.dumps(r))
        sys.exit(0 if r["ok"] else 1)
    elif len(a) >= 2 and a[0] == "features":
        print(json.dumps(features(a[1])))
    elif (len(a) >= 3 and a[0] == "build") or (len(a) >= 2 and a[0] == "check"):
        bad = locked_violation(a[1])
        if bad:
            sys.exit(bad)
        out = os.path.abspath(a[2]) if a[0] == "build" else "/nonexistent"
        if a[0] == "build":
            os.makedirs(out, exist_ok=True)
        rc = subprocess.run(["bash", "-c", BUILD, "build", out] + (["check"] if a[0] == "check" else []), cwd=a[1]).returncode
        if rc == 0:
            print("ok")
        sys.exit(rc)
    else:
        print("usage: problem.py verify <instance> <solution> | features <instance> | build <source dir> <output dir> | check <source dir>")
