#!/usr/bin/env python3
"""Critical node problem (undirected graph; remove exactly K vertices so that the pairwise connectivity of what is left
- the number of vertex pairs that are still connected, i.e. the sum over the remaining components of |C| (|C| - 1) / 2 -
is as small as possible). The solver is an open-source C++ program in one file, used as published behind a small
interface file (`host_main.cpp`, LOCKED: arguments, seeding clock, output), single thread.

Instance (`<name>.cnp`): first line n; then one line `v: neighbours` per vertex 0..n-1 (the host's native format);
then a LAST line `K <k>`. Solution: line 1 = the k removed vertices; a second line (the objective the solver reports)
is ignored. Cost = the pairwise connectivity recomputed here (an integer; smaller is better)."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

NAME = "cnp"
OBJECTIVE = "min"
INSTANCE_GLOB = "*.cnp"
TIME_LIMIT_S = 300.0

LOCKED = {"host_main.cpp": "3ede537a8b1be1b8468e386a746d1e52"}


def header(path):
    """(n, K): n from the first line, K from the last line."""
    with open(path, "rb") as f:
        n = int(f.readline().split()[0])
        f.seek(0, os.SEEK_END)
        end = f.tell()
        f.seek(max(0, end - 200))
        tail = f.read().split(b"\n")
    last = [x for x in tail if x.strip()][-1].split()
    if len(last) != 2 or last[0] != b"K":
        raise ValueError("no `K <k>` line at the end")
    return n, int(last[1])


def size(path):
    return header(path)[0]


def time_limit(n):
    return TIME_LIMIT_S


def io_slack(path):
    """Seconds allowed on top of the time limit: the host looks at its clock between two search phases, not inside
    one, and a phase takes longer on the large graphs."""
    try:
        return 10.0 + 0.002 * header(path)[0]
    except Exception:
        return 10.0


def parse(path):
    """-> (n, K, adj) with adj[v] = list of neighbours; every edge is checked to be listed from both ends later
    (validate_instance), here the lists are taken as they are."""
    n, K = header(path)
    adj = [None] * n
    with open(path, "rb") as f:
        f.readline()
        for _ in range(n):
            line = f.readline()
            a, _, b = line.partition(b":")
            v = int(a)
            if not (0 <= v < n) or adj[v] is not None:
                raise ValueError("vertex line %r out of range or repeated" % a.strip()[:20])
            adj[v] = [int(x) for x in b.split()]
    if any(x is None for x in adj):
        raise ValueError("a vertex has no line")
    return n, K, adj


def read_solution(sol_path, n):
    with open(sol_path, "rb") as f:
        first = f.readline()
    S = [int(t) for t in first.split()]
    for v in S:
        if v < 0 or v >= n:
            raise ValueError("vertex %d out of range 0..%d" % (v, n - 1))
    return S


def connectivity(n, adj, removed):
    """Pairwise connectivity of the graph without the removed vertices (iterative search over the components)."""
    gone = bytearray(n)
    for v in removed:
        gone[v] = 1
    seen = bytearray(gone)
    total = 0
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = 1
        stack, c = [s], 0
        while stack:
            u = stack.pop()
            c += 1
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = 1
                    stack.append(w)
        total += c * (c - 1) // 2
    return total


def verify(inst_path, sol_path, opts=None):
    """{"ok", "cost", "msg"}: exactly K different vertices in range; the objective is recomputed from the instance."""
    try:
        if not os.path.exists(sol_path) or os.path.getsize(sol_path) == 0:
            return {"ok": False, "cost": None, "msg": "no solution file"}
        n, K, adj = parse(inst_path)
        S = read_solution(sol_path, n)
        if len(set(S)) != len(S):
            return {"ok": False, "cost": None, "msg": "a vertex is listed twice"}
        if len(S) != K:
            return {"ok": False, "cost": None, "msg": "%d vertices removed, K = %d" % (len(S), K)}
        c = connectivity(n, adj, S)
        if c <= 0:
            return {"ok": False, "cost": None, "msg": "objective %d (no connected pair left: not a meaningful instance)" % c}
        return {"ok": True, "cost": float(c), "msg": "", "removed": len(S)}
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "verifier: %r" % (ex,)}


def validate_instance(path, lo, hi):
    try:
        n, K, adj = parse(path)
    except Exception as ex:
        return False, "unreadable: %r" % (ex,)
    if not (lo <= n <= hi):
        return False, "n = %d outside %d..%d" % (n, lo, hi)
    if not (1 <= K < n):
        return False, "K = %d" % K
    es = set()
    for v, a in enumerate(adj):
        if len(set(a)) != len(a) or v in a or any(w < 0 or w >= n for w in a):
            return False, "vertex %d: repeated neighbour, self loop or neighbour out of range" % v
        for w in a:
            es.add((v, w))
    if any((w, v) not in es for v, w in es):
        return False, "an edge is listed from one end only"
    if not es:
        return False, "no edges"
    return True, ""


def features(path):
    n, K, adj = parse(path)
    m = sum(len(a) for a in adj) // 2
    return {"n": n, "m": m, "K": K, "mean_degree": 2.0 * m / max(1, n), "k_share": K / float(n)}


def canonical(path_in, path_out, rng=None):
    import shutil
    shutil.copy(path_in, path_out)


def signature(path):
    """[n, m, K, hash of the sorted degree sequence]: equal for two instances that are the same graph with the same K
    up to a renumbering of the vertices (used by the training / test overlap check)."""
    n, K, adj = parse(path)
    deg = sorted(len(a) for a in adj)
    return [n, sum(deg) // 2, K, hashlib.md5(repr(deg).encode()).hexdigest()]


def sig_similarity(a, b):
    """1.0 when the two graphs have the same size and the same degree sequence, whatever their K (a public graph
    with another K would still be that public graph)."""
    return 1.0 if (a[0] == b[0] and a[1] == b[1] and a[3] == b[3]) else 0.0


RANGE_FEATURES = ["mean_degree", "k_share"]

# Build: one translation unit (host_main.cpp includes the solver file), fixed flags, then the launcher.
BUILD = r'''set -e
set -o pipefail
CXX=${CXX:-g++}
command -v "$CXX" > /dev/null || { echo "no C++ compiler ($CXX) on PATH"; exit 1; }
[ -f host_main.cpp ] || { echo "host_main.cpp is missing"; exit 1; }
[ -f solver.cpp ] || { echo "solver.cpp is missing"; exit 1; }
if [ "$2" = check ]; then                   # compile errors only, no binary
  "$CXX" -std=gnu++17 -O1 -w -fsyntax-only host_main.cpp 2>&1 | tail -n 150
  exit 0
fi
"$CXX" -std=gnu++17 -O3 -march=native -w host_main.cpp -o "$1/prog" 2>&1 | tail -n 150
cat > "$1/solver" <<'LAUNCHER'
#!/usr/bin/env python3
"""solver <instance> <solution> -t <seconds> -seed <int>: one run of the program on one thread."""
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
inst, sol = a[0], a[1]
T = a[a.index("-t") + 1]
seed = a[a.index("-seed") + 1]
cmd = [os.path.join(here, "prog"), inst, T, seed, sol]
os.execv(cmd[0], cmd)
LAUNCHER
chmod +x "$1/solver"
'''


def run_cmd(bindir, inst_path, sol_path, T, seed, opts=None):
    return [os.path.join(bindir, "solver"), inst_path, sol_path, "-t", "%d" % max(1, int(T)), "-seed", str(seed)]


_LINE = re.compile(r"^RESULT n \d+ k \d+ objective (\d+) best_time ([0-9.]+) ")


def trace(sol_path, stdout_path):
    """[[seconds, cost]]: the program prints one result line (objective, time of the best solution)."""
    try:
        for line in open(stdout_path, errors="ignore"):
            m = _LINE.match(line.strip())
            if m:
                return [[float(m.group(2)), float(m.group(1))]]
    except OSError:
        pass
    return []


def locked_violation(tree):
    """None, or why the tree is refused: the interface file differs from the original or is missing."""
    tree = Path(tree)
    for rel, md5 in LOCKED.items():
        p = tree / rel
        if not p.exists():
            return "BUILD REFUSED: %s is missing; it must stay exactly as in the original." % rel
        if hashlib.md5(p.read_bytes()).hexdigest() != md5:
            return ("BUILD REFUSED: %s differs from the original. This file (arguments, seeding clock and output of the "
                    "program) must stay byte-identical; put your changes in the solver's own file(s)." % rel)
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
