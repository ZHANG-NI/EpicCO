#!/usr/bin/env python3
"""Final evaluation: several solvers ("arms") on several instance suites with new seeds, every run on a pinned core,
every solution re-verified, every run's anytime trace stored.   python3 final_eval.py <spec.json>

spec = {
  "problem": "cvrp", "out": "<dir>", "prio": 6,
  "arms":   {"<name>": {"tree": "<source tree built with the plugin's BUILD>"}
                     | {"bindir": "<dir with an executable `solver` following the plugin's run_cmd>"}
                     | {"cmd": ["...", "{inst}", "{sol}", "{T}", "{seed}"], "trace": "plugin"|"stdout"}},
  "suites": {"<name>": {"dir": "<instances>", "bks": "<json name->value>"|null, "opts": {}, "seeds": [11, 12, 13],
                        "time_scale": 1.0, "min_n": 0, "max_n": 10**9}}
}
Resumable: finished (arm, suite, instance, seed) runs are skipped. Results: <out>/results.jsonl, one JSON per run:
{"arm","suite","name","n","seed","T","elapsed","ok","cost","msg","trace":[[t,cost],...]}.
"""
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

FW = Path(__file__).resolve().parent
sys.path.insert(0, str(FW))
import poolclient   # noqa: E402

SE = os.environ.get("SE_ROOT", "/opt/epicco")


def stdout_trace(path):
    out = []
    try:
        for line in open(path, encoding="utf-8", errors="ignore"):
            if line.startswith("TRACE "):
                q = line.split()
                try:
                    out.append((float(q[1]), float(q[2])))
                except (ValueError, IndexError):
                    pass
    except OSError:
        pass
    return out


def thin(tr, k=200):
    if len(tr) <= k:
        return tr
    step = len(tr) / float(k)
    return [tr[int(i * step)] for i in range(k)] + [tr[-1]]


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    P = importlib.import_module("problems." + spec["problem"])
    IO = getattr(P, "io_slack", None) or (lambda p: 0.0)    # seconds on top of the time limit for reading the instance
    out = Path(spec["out"])
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "bin").mkdir(exist_ok=True)
    resf = out / "results.jsonl"
    done = set()
    if resf.exists():
        for line in open(resf):
            try:
                r = json.loads(line)
                done.add((r["arm"], r["suite"], r["name"], r["seed"]))
            except Exception:
                pass
    env = dict(os.environ)
    env["PATH"] = (spec.get("toolchain_bin") or os.environ.get("SE_TOOLCHAIN_BIN", "/opt/epicco/toolchain/bin")) + ":" + env.get("PATH", "")
    arms = {}
    for name, a in spec["arms"].items():
        if "tree" in a:
            h = hashlib.sha1(str(Path(a["tree"]).resolve()).encode()).hexdigest()[:10]
            b = out / "bin" / (name + "_" + h)
            if not (b / "solver").exists():
                b.mkdir(parents=True, exist_ok=True)
                r = subprocess.run(["bash", "-c", P.BUILD, "build", str(b)], cwd=a["tree"], env=env, capture_output=True, text=True)
                if r.returncode != 0:
                    sys.exit("build of arm %s failed:\n%s" % (name, (r.stdout + r.stderr)[-3000:]))
            arms[name] = {"bindir": str(b)}
        else:
            arms[name] = a
    jobs, meta = [], []
    for sname, s in spec["suites"].items():
        files = sorted(Path(s["dir"]).glob(P.INSTANCE_GLOB))
        for f in files:
            try:
                n = P.size(str(f))
            except Exception:
                continue
            if n < s.get("min_n", 0) or n > s.get("max_n", 10**9):
                continue
            if s.get("only") and f.stem not in s["only"]:
                continue
            T = P.time_limit(n) * float(s.get("time_scale", 1.0))
            for aname, a in arms.items():
                if s.get("arms") and aname not in s["arms"]:
                    continue
                for seed in s["seeds"]:
                    if (aname, sname, f.stem, seed) in done:
                        continue
                    base = out / "runs" / ("%s__%s__%s__s%d" % (aname, sname, f.stem, seed))
                    sol, log = str(base) + ".sol", str(base) + ".log"
                    if "cmd" in a:
                        cmd = [x.replace("{inst}", str(f)).replace("{sol}", sol).replace("{T}", "%.2f" % T).replace("{seed}", str(seed))
                               .replace("{n}", str(n)) for x in a["cmd"]]
                    else:
                        cmd = P.run_cmd(a["bindir"], str(f), sol, T, seed, s.get("opts", {}))
                    jobs.append({"cmd": cmd, "cwd": str(out / "runs"), "timeout": T * float(a.get("timeout_factor", 1.0)) + 180 + IO(str(f)), "stdout": log,
                                 "env": a.get("env", {})})
                    meta.append((aname, sname, f, n, seed, T, sol, log, s.get("opts", {}), a))
    first = list(spec.get("suites_first", []))                       # these suites are finished before the others start
    rank = lambda i: 0 if meta[i][1] in first else 1                 # noqa: E731
    order = sorted(range(len(jobs)), key=lambda i: (rank(i), -meta[i][5]))      # then longest first
    jobs, meta = [jobs[i] for i in order], [meta[i] for i in order]
    print("runs to do: %d (already done: %d), core-seconds: %.0f" % (len(jobs), len(done), sum(m[5] for m in meta)), flush=True)
    CH = int(spec.get("chunk", 256))
    for c in range(0, len(jobs), CH):
        res = poolclient.run_jobs(spec.get("pool", SE + "/pool"), jobs[c:c + CH], prio=int(spec.get("prio", 6)),
                                  tag=spec.get("tag", "final"))
        with open(resf, "a") as fo:
            for (aname, sname, f, n, seed, T, sol, log, opts, a), r in zip(meta[c:c + CH], res):
                v = P.verify(str(f), sol, opts)
                slack = 5.0 + 0.03 * T + IO(str(f))
                over = r["timed_out"] or r["elapsed"] > T * float(a.get("timeout_factor", 1.0)) + slack
                ok = bool(v["ok"]) and not (over and not a.get("allow_overtime"))
                cpu = r.get("cpu_s")
                multi = cpu is not None and cpu > 1.03 * r["elapsed"] + 2.0          # one instance, one core
                ok = ok and not multi
                tr = P.trace(sol, log) if a.get("trace", "plugin") == "plugin" else stdout_trace(log)
                fo.write(json.dumps({"arm": aname, "suite": sname, "name": f.stem, "n": n, "seed": seed, "T": T,
                                     "elapsed": round(r["elapsed"], 2), "ok": ok, "cost": v["cost"] if v["ok"] else None,
                                     "overtime": bool(over), "multi_core": bool(multi), "cpu_s": cpu, "msg": v.get("msg", ""),
                                     "trace": thin(tr)}) + "\n")
        print("chunk %d/%d done" % (c // CH + 1, (len(jobs) + CH - 1) // CH), flush=True)
    print("FINAL_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
