#!/usr/bin/env python3
"""Maximum weight independent set (vertex-weighted graph; choose pairwise non-adjacent vertices of maximum total
weight). The solver is an open-source C program, used as published and run SEQUENTIALLY on one core (its own option
`-c 1`) from a cold start (no initial solution).

Instance: METIS text (`n m 10`, then per vertex its weight and its 1-based neighbours) = the host's native format.
Next to every `<name>.graph` the tools that create instances write three raw binary files (`.w64` int64 weights,
`.x64` int64 offsets, `.a32` int32 neighbours): the verifier reads those (memory-mapped) and never the
solver's own parser; without them it parses the text (small instances only).
Solution: one 1-based vertex per line.
Cost: the framework minimises, so cost = 1e15 / (total weight of the set): a cost difference of -0.1 % is a weight
gain of 0.1 % (to first order), no reference value is needed, equal weights give equal costs.

One core: the host's algorithm files carry OpenMP directives. They stay (SCAN_ALLOW), the launcher runs the binary
with OMP_NUM_THREADS=1, OMP_THREAD_LIMIT=1 and `-c 1`, `src/main.c` (options, thread count, timing of the search,
output) is LOCKED to the original, and calls that set a thread count are refused anywhere else."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

NAME = "mwis"
OBJECTIVE = "min"
INSTANCE_GLOB = "*.graph"
TIME_LIMIT_S = 360.0
COST_SCALE = 1e15
SIDE_EXT = (".w64", ".x64", ".a32")      # binary files of the verifier next to <name>.graph (see parse)

LOCKED = {"src/main.c": "28daf425b3cf0c5a26d913525a7d2851"}
SCAN_ALLOW = r"#\s*include\s*<omp\.h>|#\s*pragma\s+omp\b"
_FORBID = re.compile(r"omp_set_num_threads|\bnum_threads\s*\(|omp_set_dynamic|omp_set_nested|omp_set_max_active_levels|"
                     r"omp_set_schedule|\bsetenv\s*\(|\bputenv\s*\(")
PARAM_OK = re.compile(r"^(-p \d{1,4}|-s \d+(\.\d+)?|-q \d{1,6})$")


def header(path):
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"%"):
                continue
            p = line.split()
            return int(p[0]), int(p[1])
    raise ValueError("empty instance")


def size(path):
    return header(path)[0]


def time_limit(n):
    return TIME_LIMIT_S


def io_slack(path):
    """Seconds allowed on top of the time limit for reading the graph: the host's time limit is its search time, and a
    text graph of several gigabytes takes minutes to read. 30 s + 150 s per gigabyte of the instance file."""
    try:
        return 30.0 + 150.0 * os.path.getsize(path) / 1e9
    except OSError:
        return 30.0


def _sidecar(path):
    return [str(path) + e for e in SIDE_EXT]


def _map(p, fmt):
    import mmap
    if os.path.getsize(p) == 0:
        return memoryview(b"").cast(fmt)
    with open(p, "rb") as f:
        return memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)).cast(fmt)


def parse(path):
    """-> (n, weights, xadj, adj) with 0-based neighbours: memory-mapped views of the binary files when they exist
    (raw little-endian int64 weights, int64 offsets, int32 neighbours), else lists parsed from the text."""
    sc = _sidecar(path)
    if all(os.path.exists(p) for p in sc):
        w, xadj, adj = _map(sc[0], "q"), _map(sc[1], "q"), _map(sc[2], "i")
        if len(xadj) != len(w) + 1 or xadj[len(w)] != len(adj) or len(w) != header(path)[0]:
            raise ValueError("binary files do not fit the instance")
        return len(w), w, xadj, adj
    n, m = header(path)
    w, xadj, adj = [], [0], []
    with open(path, "rb") as f:
        first = True
        for line in f:
            if line.startswith(b"%"):
                continue
            if first:
                first = False
                continue
            if len(w) >= n:
                break
            p = line.split()
            w.append(int(p[0]) if p else 1)
            adj.extend(int(x) - 1 for x in p[1:])
            xadj.append(len(adj))
    while len(w) < n:                       # trailing vertices without a line
        w.append(1)
        xadj.append(len(adj))
    return n, w, xadj, adj


def read_solution(sol_path, n):
    S = []
    with open(sol_path, "rb") as f:
        for tok in f.read().split():
            v = int(tok)
            if v < 1 or v > n:
                raise ValueError("vertex %d out of range 1..%d" % (v, n))
            S.append(v - 1)
    return S


def verify(inst_path, sol_path, opts=None):
    """{"ok", "cost", "msg", "weight", "size"}: every vertex in range and listed once, no two chosen vertices adjacent;
    the weight is summed here from the instance's weights."""
    try:
        if not os.path.exists(sol_path) or os.path.getsize(sol_path) == 0:
            return {"ok": False, "cost": None, "msg": "no solution file"}
        n, w, xadj, adj = parse(inst_path)
        S = read_solution(sol_path, n)
        if not S:
            return {"ok": False, "cost": None, "msg": "empty set"}
        if len(set(S)) != len(S):
            return {"ok": False, "cost": None, "msg": "a vertex is listed twice"}
        chosen = set(S)
        mapped = isinstance(adj, memoryview)
        for u in S:
            a, b = xadj[u], xadj[u + 1]
            if b > a and not chosen.isdisjoint(adj[a:b].tolist() if mapped else adj[a:b]):
                return {"ok": False, "cost": None, "msg": "two chosen vertices are adjacent (vertex %d)" % (u + 1)}
        W = sum(w[u] for u in S)
        if W <= 0:
            return {"ok": False, "cost": None, "msg": "weight %d" % W}
        return {"ok": True, "cost": COST_SCALE / W, "msg": "", "weight": W, "size": len(S)}
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "verifier: %r" % (ex,)}


def weight_of(cost):
    """Total weight behind a cost value (exact for weights below 2^52)."""
    return int(round(COST_SCALE / cost))


def validate_instance(path, lo, hi):
    try:
        n, m = header(path)
    except Exception as ex:
        return False, "unreadable: %r" % (ex,)
    if not (lo <= n <= hi):
        return False, "n = %d outside %d..%d" % (n, lo, hi)
    if m <= 0:
        return False, "no edges"
    return True, ""


def features(path):
    n, m = header(path)
    return {"n": n, "m": m, "mean_degree": 2.0 * m / max(1, n)}


def canonical(path_in, path_out, rng=None):
    import shutil
    shutil.copy(path_in, path_out)
    for a, b in zip(_sidecar(path_in), _sidecar(path_out)):
        if os.path.exists(a):
            shutil.copy(a, b)


def signature(path):
    """[n, m, hash of the sorted degree sequence and the sorted weights]: equal for two instances that are the same
    graph with the same weights up to a renumbering of the vertices (used by the training / test overlap check)."""
    n, w, x, a = parse(path)
    deg = sorted(x[i + 1] - x[i] for i in range(n))
    ws = sorted(w[i] for i in range(n))
    return [n, len(a) // 2, hashlib.md5((repr(deg) + "|" + repr(ws)).encode()).hexdigest()]


def sig_similarity(a, b):
    return 1.0 if list(a) == list(b) else 0.0


RANGE_FEATURES = ["mean_degree"]

# Build: the command-line program only (the library wrapper chils.c is not part of it), fixed flags (the published
# Makefile's), then the launcher. `params.txt` (one line, optional) may hold the host's own tuning options -p, -s, -q.
BUILD = r'''set -e
set -o pipefail
CC=${CC:-gcc}
command -v "$CC" > /dev/null || { echo "no C compiler ($CC) on PATH"; exit 1; }
[ -f src/main.c ] || { echo "src/main.c is missing"; exit 1; }
touch params.txt
python3 - <<'PY' || exit 1
import re, sys
t = open("params.txt").read().split()
ok = re.compile(r"^(-p \d{1,4}|-s \d+(\.\d+)?|-q \d{1,6})$")
bad = [" ".join(t[i:i + 2]) for i in range(0, len(t), 2) if not ok.match(" ".join(t[i:i + 2]))]
if bad or len(t) % 2:
    sys.exit("params.txt may only hold the options -p <int>, -s <seconds>, -q <int> (found: %s)" % (bad or t))
PY
SRCS=$(ls src/*.c | grep -v '^src/chils\.c$')
if [ "$2" = check ]; then                   # compile errors only, no binary
  "$CC" -std=gnu17 -O1 -I include -fopenmp -fsyntax-only $SRCS 2>&1 | tail -n 150
  exit 0
fi
"$CC" -std=gnu17 -O3 -march=native -I include -fopenmp -DNDEBUG $SRCS -o "$1/prog" 2>&1 | tail -n 150
cp params.txt "$1/params.txt"
cat > "$1/solver" <<'LAUNCHER'
#!/usr/bin/env python3
"""solver <instance> <solution> -t <seconds> -seed <int>: runs the host on ONE thread (its option -c 1, and the OpenMP
runtime limited to one thread) from a cold start; the tuning options of params.txt (-p, -s, -q) are passed on."""
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
inst, sol = a[0], a[1]
T = a[a.index("-t") + 1]
seed = a[a.index("-seed") + 1]
extra = open(os.path.join(here, "params.txt")).read().split()
env = dict(os.environ)
env.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_DYNAMIC="false", OMP_PROC_BIND="false")
cmd = [os.path.join(here, "prog"), "-g", inst, "-o", sol, "-t", T, "-r", seed, "-c", "1"] + extra
os.execve(cmd[0], cmd, env)
LAUNCHER
chmod +x "$1/solver"
'''


def run_cmd(bindir, inst_path, sol_path, T, seed, opts=None):
    return [os.path.join(bindir, "solver"), inst_path, sol_path, "-t", "%d" % max(1, int(T)), "-seed", str(seed)]


_LINE = re.compile(r"^[^,\s]+,\d+,\d+,(\d+),([0-9.]+),([0-9.]+)\s*$")


def trace(sol_path, stdout_path):
    """[[seconds, cost]]: the host prints one result line (weight, time of the best solution, total time)."""
    try:
        for line in open(stdout_path, errors="ignore"):
            m = _LINE.match(line.strip())
            if m and int(m.group(1)) > 0:
                return [[float(m.group(2)), COST_SCALE / int(m.group(1))]]
    except OSError:
        pass
    return []


def locked_violation(tree):
    """None, or why the tree is refused: a LOCKED file differs from the original, or a call that sets a thread count /
    changes the environment appears outside the locked file."""
    tree = Path(tree)
    for rel, md5 in LOCKED.items():
        p = tree / rel
        if not p.exists():
            return "BUILD REFUSED: %s is missing; it must stay exactly as in the original." % rel
        if hashlib.md5(p.read_bytes()).hexdigest() != md5:
            return ("BUILD REFUSED: %s differs from the original. This file (options, thread count, timing and output of "
                    "the program) must stay byte-identical; put your changes in the other files." % rel)
    for p in sorted(tree.rglob("*")):
        rel = p.relative_to(tree).as_posix()
        if rel in LOCKED or not p.is_file() or p.suffix.lower() not in (".c", ".h"):
            continue
        for i, line in enumerate(p.read_text(errors="ignore").split("\n"), 1):
            m = _FORBID.search(line)
            if m:
                return ("BUILD REFUSED: the solver must run as ONE thread on ONE core. Found `%s` in %s line %d. Setting a "
                        "thread count or changing the environment is not allowed (the existing `#pragma omp` lines may stay: "
                        "the program is run with a thread limit of one)." % (m.group(0), rel, i))
    return None


if __name__ == "__main__":
    import subprocess
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "verify":
        r = verify(a[1], a[2])
        print(json.dumps(r))
        sys.exit(0 if r["ok"] else 1)
    elif len(a) >= 2 and a[0] == "features":
        print(json.dumps(features(a[1])))
    elif (len(a) >= 3 and a[0] == "build") or (len(a) >= 2 and a[0] == "check"):
        # your own build / compile check of a source tree, with the official commands
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
