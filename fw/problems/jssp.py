"""Job-shop scheduling plugin (classical JSP, makespan): instance parsing, independent solution checking, build and
run commands.

Instance file: first the number of jobs n and of machines m, then one line
per job with m pairs `machine duration` (machines numbered from 0) in the order in which the job visits them. Every
job visits every machine exactly once.

Solution file (written by the solver): first line `n m makespan`, then m lines; line k lists the n jobs (numbered
from 1) in the order in which machine k-1 processes them.

The check is independent of the solver: it rebuilds the earliest-start schedule from the machine sequences (longest
path in the disjunctive graph), rejects a cyclic selection (infeasible) and recomputes the makespan."""
import os

NAME = "jssp"
OBJECTIVE = "min"
INSTANCE_GLOB = "*.txt"
TIME_LIMIT_S = 300.0


def parse(path):
    tok = open(path, encoding="utf-8", errors="ignore").read().split()
    n, m = int(tok[0]), int(tok[1])
    v = [int(x) for x in tok[2:2 + 2 * n * m]]
    if len(v) != 2 * n * m:
        raise ValueError("expected %d numbers after the header, found %d" % (2 * n * m, len(v)))
    mach = [[v[2 * (j * m + k)] for k in range(m)] for j in range(n)]
    dur = [[v[2 * (j * m + k) + 1] for k in range(m)] for j in range(n)]
    for j in range(n):
        if sorted(mach[j]) != list(range(m)):
            raise ValueError("job %d does not visit every machine exactly once" % (j + 1))
    return n, m, mach, dur


def size(path):
    """Number of operations (jobs x machines): the size used for size bands and reports."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            q = line.split()
            if len(q) >= 2:
                return int(q[0]) * int(q[1])
    raise ValueError("empty file")


def shape(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            q = line.split()
            if len(q) >= 2:
                return int(q[0]), int(q[1])
    raise ValueError("empty file")


def time_limit(n):
    """Five minutes for every instance, whatever its size."""
    return TIME_LIMIT_S


def lower_bound(n, m, mach, dur):
    """Trivial bound: the longest job and the most loaded machine."""
    lb = max(sum(d) for d in dur)
    load = [0] * m
    for j in range(n):
        for k in range(m):
            load[mach[j][k]] += dur[j][k]
    return max(lb, max(load))


def makespan_of(n, m, mach, dur, seq):
    """seq[k] = jobs (0-based) in processing order on machine k. Returns the makespan of the earliest-start schedule,
    or None when the machine sequences contradict the job routes (a cycle)."""
    pos = [[0] * m for _ in range(n)]            # position of machine k in the route of job j
    for j in range(n):
        for k in range(m):
            pos[j][mach[j][k]] = k
    nxt_m = {}                                    # machine successor of operation (j, k)
    indeg = [[0] * m for _ in range(n)]
    for mc in range(m):
        s = seq[mc]
        for a in range(n - 1):
            u = (s[a], pos[s[a]][mc])
            w = (s[a + 1], pos[s[a + 1]][mc])
            nxt_m[u] = w
            indeg[w[0]][w[1]] += 1
    for j in range(n):
        for k in range(1, m):
            indeg[j][k] += 1
    end = [[0] * m for _ in range(n)]
    start = [[0] * m for _ in range(n)]
    stack = [(j, k) for j in range(n) for k in range(m) if indeg[j][k] == 0]
    done = 0
    while stack:
        j, k = stack.pop()
        done += 1
        e = start[j][k] + dur[j][k]
        end[j][k] = e
        succ = []
        if k + 1 < m:
            succ.append((j, k + 1))
        w = nxt_m.get((j, k))
        if w is not None:
            succ.append(w)
        for (a, b) in succ:
            if e > start[a][b]:
                start[a][b] = e
            indeg[a][b] -= 1
            if indeg[a][b] == 0:
                stack.append((a, b))
    if done != n * m:
        return None
    return max(end[j][m - 1] for j in range(n))


def read_solution(sol_path):
    lines = [l.split() for l in open(sol_path, encoding="utf-8", errors="ignore").read().splitlines() if l.strip()]
    n, m = int(lines[0][0]), int(lines[0][1])
    claimed = lines[0][2] if len(lines[0]) > 2 else None
    seq = [[int(x) - 1 for x in lines[1 + k]] for k in range(m)]
    return n, m, claimed, seq


def verify(inst_path, sol_path, opts=None):
    try:
        n, m, mach, dur = parse(inst_path)
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "instance: %r" % (ex,)}
    if not os.path.exists(sol_path):
        return {"ok": False, "cost": None, "msg": "no solution file"}
    try:
        sn, sm, claimed, seq = read_solution(sol_path)
    except Exception as ex:
        return {"ok": False, "cost": None, "msg": "bad solution file: %r" % (ex,)}
    if sn != n or sm != m or len(seq) != m:
        return {"ok": False, "cost": None, "msg": "solution is for %s x %s, instance is %d x %d" % (sn, sm, n, m)}
    for k in range(m):
        if sorted(seq[k]) != list(range(n)):
            return {"ok": False, "cost": None, "msg": "machine %d: not a permutation of the jobs 1..%d" % (k, n)}
    c = makespan_of(n, m, mach, dur, seq)
    if c is None:
        return {"ok": False, "cost": None, "msg": "the machine sequences contradict the job routes (cycle)"}
    if c < lower_bound(n, m, mach, dur):
        return {"ok": False, "cost": None, "msg": "makespan below the trivial lower bound (checker error)"}
    return {"ok": True, "cost": c, "msg": "" if claimed is None or str(claimed) == str(c) else "solver reported %s" % claimed}


def validate_instance(path, lo, hi):
    try:
        n, m, mach, dur = parse(path)
    except Exception as ex:
        return False, "parse: %r" % (ex,)
    if n * m < lo or n * m > hi:
        return False, "n*m=%d outside [%d, %d]" % (n * m, lo, hi)
    if min(min(r) for r in dur) < 1:
        return False, "a duration below 1"
    if max(max(r) for r in dur) > 10**4:
        return False, "a duration above 10^4"
    return True, ""


def features(path):
    n, m, mach, dur = parse(path)
    flat = [x for r in dur for x in r]
    half = m // 2
    # share of jobs whose first half of the route stays inside the first half of the machines (two-stage job shops)
    two_set = sum(1 for j in range(n) if all(x < half for x in mach[j][:half])) / float(n)
    return {"n": n * m, "jobs": n, "machines": m, "ratio": round(n / float(m), 3), "dur_min": min(flat), "dur_max": max(flat),
            "dur_mean": round(sum(flat) / float(len(flat)), 2), "two_set_share": round(two_set, 3)}


def canonical(path_in, path_out, rng=None):
    """Re-write an instance in one fixed layout. With `rng`: also relabel the machines and reorder the jobs by random
    permutations: the same problem, no longer recognisable as a particular file."""
    n, m, mach, dur = parse(path_in)
    if rng is not None:
        pm = list(range(m))
        pj = list(range(n))
        rng.shuffle(pm)
        rng.shuffle(pj)
        mach = [[pm[x] for x in mach[j]] for j in pj]
        dur = [dur[j] for j in pj]
    with open(path_out, "w") as f:
        f.write("%d %d\n" % (n, m))
        for j in range(n):
            f.write(" ".join("%d %d" % (mach[j][k], dur[j][k]) for k in range(m)) + "\n")


def signature(path):
    n, m, mach, dur = parse(path)
    return (n, m, tuple(sorted(tuple(sorted(r)) for r in dur)))


def sig_similarity(a, b):
    """1.0 when instance a has the same shape and the same multiset of per-job duration multisets as instance b (a
    relabelled copy), else 0.0."""
    return 1.0 if a == b else 0.0


# features of a generated instance that must stay inside the public benchmark's range
RANGE_FEATURES = ["ratio", "dur_mean"]

BUILD = r'''set -e
g++ -O3 -DNDEBUG -std=gnu++17 -w *.cpp -o "$1/solver"
'''


def run_cmd(bindir, inst_path, sol_path, T, seed, opts=None):
    return [os.path.join(bindir, "solver"), inst_path, sol_path, "-t", "%.2f" % T, "-seed", str(seed)]


def trace(sol_path, stdout_path):
    out = []
    if os.path.exists(stdout_path):
        for line in open(stdout_path, encoding="utf-8", errors="ignore"):
            if line.startswith("TRACE "):
                q = line.split()
                try:
                    out.append((float(q[1]), float(q[2])))
                except (ValueError, IndexError):
                    pass
    return out


if __name__ == "__main__":
    import json
    import sys
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "verify":
        print(json.dumps(verify(a[1], a[2])))
    elif len(a) >= 2 and a[0] == "validate":
        print(validate_instance(a[1], int(a[2]) if len(a) > 2 else 1, int(a[3]) if len(a) > 3 else 10**6))
    elif len(a) >= 2 and a[0] == "features":
        print(json.dumps(features(a[1])))
    else:
        print("usage: problem.py verify <instance> <solution> | validate <instance> [min_n max_n] | features <instance>")
