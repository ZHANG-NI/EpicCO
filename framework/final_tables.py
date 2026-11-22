#!/usr/bin/env python3
"""final_tables.py <final_eval out dir>: tables and anytime curves from results.jsonl.

Per suite and arm: mean gap (%) to the reference (best-known value when the suite has one, otherwise the best cost
any arm reached), its standard deviation over seeds, the number of instances at the reference (mean over seeds),
failed runs, and paired wins/losses against the arm `original` on the per-instance mean cost.
Curves: mean gap as a function of the fraction of the time limit used (from every run's trace).
Writes <out>/tables.md and <out>/curves.json."""
import json
import sys
from collections import defaultdict
from pathlib import Path

GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def scan(out, spec, rows):
    """--scan: add the runs that are finished on disk (solution file older than 30 s) but not yet in results.jsonl
    (final_eval.py records per chunk). Preliminary: cost re-verified, time limit and CPU time NOT checked."""
    import importlib
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    P = importlib.import_module("problems." + spec["problem"])
    ext = P.INSTANCE_GLOB.replace("*", "")
    have = {(r["arm"], r["suite"], r["name"], r["seed"]) for r in rows}
    add = 0
    for sol in sorted((out / "runs").glob("*.sol")):
        q = sol.stem.split("__")
        if len(q) != 4 or q[1] not in spec["suites"] or not q[3].startswith("s"):
            continue
        key = (q[0], q[1], q[2], int(q[3][1:]))
        if key in have or time.time() - sol.stat().st_mtime < 30:
            continue
        s = spec["suites"][q[1]]
        inst = Path(s["dir"]) / (q[2] + ext)
        v = P.verify(str(inst), str(sol), s.get("opts", {}))
        rows.append({"arm": q[0], "suite": q[1], "name": q[2], "n": P.size(str(inst)), "seed": key[3], "T": None,
                     "ok": bool(v["ok"]), "cost": v["cost"] if v["ok"] else None, "trace": []})
        add += 1
    return add


def paired(rs, a, b, names, pred=None):
    """Arm a against arm b: per instance, the relative cost difference (% of b) averaged over the seeds both have;
    mean over instances, standard error, t, wins/losses. Negative = a is better."""
    ca, cb = {}, {}
    for r in rs:
        if r["ok"] and r["arm"] in (a, b):
            (ca if r["arm"] == a else cb)[(r["name"], r["seed"])] = (r["cost"], r["n"])
    by = defaultdict(list)
    for k, (x, n) in ca.items():
        if k in cb and k[0] in names and (pred is None or pred(n)):
            by[k[0]].append(100.0 * (x - cb[k][0]) / cb[k][0])
    d = [sum(v) / len(v) for v in by.values()]
    if len(d) < 2:
        return None
    m = sum(d) / len(d)
    se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5
    return {"n": len(d), "mean": m, "se": se, "t": m / se if se > 0 else 0.0,
            "wins": sum(x < -1e-12 for x in d), "losses": sum(x > 1e-12 for x in d)}


def main():
    out = Path(sys.argv[1])
    spec = json.loads((out / "spec.json").read_text())
    rows = [json.loads(l) for l in open(out / "results.jsonl")]
    L, curves = [], {}
    if "--scan" in sys.argv:
        L.append("PRELIMINARY: %d finished runs read from disk, not yet recorded (time limit / CPU time unchecked)" % scan(out, spec, rows))
    for sname, s in spec["suites"].items():
        rs = [r for r in rows if r["suite"] == sname]
        if not rs:
            continue
        bks = json.loads(Path(s["bks"]).read_text()) if s.get("bks") else {}
        ref = {}
        for r in rs:
            if r["ok"]:
                ref[r["name"]] = min(ref.get(r["name"], float("inf")), r["cost"])
        kind = "best-known value" if bks else "best cost found by any arm"
        for k in list(ref):
            if k in bks:
                ref[k] = bks[k]
        arms = [a for a in spec["arms"] if any(r["arm"] == a for r in rs)]
        seeds = sorted({r["seed"] for r in rs})
        names = sorted(ref)
        L += ["", "## %s  (%d instances, seeds %s, reference = %s)" % (sname, len(names), seeds, kind), "",
              "| arm | mean gap %% | sd over seeds | at reference (of %d) | below reference | failed runs | wins / losses vs original |" % len(names),
              "|---|---|---|---|---|---|---|"]
        mean_cost = defaultdict(dict)
        for a in arms:
            per_seed, hits, below, fails = [], [], 0, 0
            for sd in seeds:
                g, h = [], 0
                for r in rs:
                    if r["arm"] != a or r["seed"] != sd or r["name"] not in ref:
                        continue
                    if not r["ok"]:
                        fails += 1
                        continue
                    x = 100.0 * (r["cost"] - ref[r["name"]]) / ref[r["name"]]
                    g.append(x)
                    h += x <= 1e-9
                    below += x < -1e-9
                if g:
                    per_seed.append(sum(g) / len(g))
                    hits.append(h)
            for nm in names:
                c = [r["cost"] for r in rs if r["arm"] == a and r["name"] == nm and r["ok"]]
                if c:
                    mean_cost[a][nm] = sum(c) / len(c)
            mu = sum(per_seed) / len(per_seed) if per_seed else float("nan")
            sd_ = (sum((x - mu) ** 2 for x in per_seed) / (len(per_seed) - 1)) ** 0.5 if len(per_seed) > 1 else 0.0
            wl = ""
            if a != "original" and "original" in mean_cost:
                both = [nm for nm in names if nm in mean_cost[a] and nm in mean_cost["original"]]
                wl = "%d / %d" % (sum(mean_cost[a][nm] < mean_cost["original"][nm] - 1e-9 for nm in both),
                                  sum(mean_cost[a][nm] > mean_cost["original"][nm] + 1e-9 for nm in both))
            L.append("| %s | %.4f | %.4f | %.1f | %d | %d | %s |" % (a, mu, sd_, sum(hits) / len(hits) if hits else 0, below, fails, wl))
            cur = []
            for tau in GRID:
                g = []
                for r in rs:
                    if r["arm"] != a or not r["ok"] or r["name"] not in ref or not r.get("trace"):
                        continue
                    pts = [c for t, c in r["trace"] if t <= tau * r["T"]]
                    if pts:
                        g.append(100.0 * (min(pts) - ref[r["name"]]) / ref[r["name"]])
                cur.append([tau, (sum(g) / len(g)) if g else None, len(g)])
            curves.setdefault(sname, {})[a] = cur
        # paired comparisons: every arm against the original, plus the pairs listed in the spec (e.g. a candidate
        # against the version it was derived from); all instances, and the smaller / larger half by size
        ns = sorted({r["n"] for r in rs})
        med = ns[len(ns) // 2]
        prs = [(a, "original") for a in arms if a != "original" and "original" in arms]
        prs += [(a, b) for a, b in spec.get("pairs", []) if a in arms and b in arms]
        if prs:
            L += ["", "| paired: A vs B | instances | runs counted per arm | mean cost diff %% | SE | t | A better / worse | "
                  "n < %d: diff (t) | n >= %d: diff (t) |" % (med, med), "|---|---|---|---|---|---|---|---|---|"]
        for a, b in prs:
            p = paired(rs, a, b, names)
            if not p:
                continue
            lo, hi = paired(rs, a, b, names, lambda n: n < med), paired(rs, a, b, names, lambda n: n >= med)
            cnt = "%d / %d" % (sum(r["arm"] == a and r["ok"] for r in rs), sum(r["arm"] == b and r["ok"] for r in rs))
            f = lambda q: ("%+.4f (%+.2f)" % (q["mean"], q["t"])) if q else "n/a"   # noqa: E731
            L.append("| %s vs %s | %d | %s | %+.4f | %.4f | %+.2f | %d / %d | %s | %s |" % (
                a, b, p["n"], cnt, p["mean"], p["se"], p["t"], p["wins"], p["losses"], f(lo), f(hi)))
    if "--scan" in sys.argv:
        (out / "tables_prelim.md").write_text("\n".join(L) + "\n")
    else:
        (out / "tables.md").write_text("\n".join(L) + "\n")
        (out / "curves.json").write_text(json.dumps(curves))
    print("\n".join(L))


if __name__ == "__main__":
    main()
