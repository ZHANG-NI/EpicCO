#!/usr/bin/env python3
"""Judge of one campaign. The only component that builds, runs, verifies, scores and commits solver versions.

  python3 judge.py <camp_dir> serve      request loop for the two agents (enhancer box, instance-agent box)
  python3 judge.py <camp_dir> calib      build v0, run it on the training benchmark with 3 seeds, set the noise levels
  python3 judge.py <camp_dir> seal <k>   generate k sealed instances from the last accepted generator (after the session)

Agents talk to it only through request/response files in their own workspace (tools/ctl.py). Everything that decides
anything (reference values, cached results, committed versions, the ledger, generator seeds) lives in <camp>/priv,
which no agent can see.
"""
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

FW = Path(__file__).resolve().parent
sys.path.insert(0, str(FW))
import jail          # noqa: E402
import poolclient    # noqa: E402

SKIP_DIRS = {"build", "bin", "obj", "target", "__pycache__", ".git"}


def tree_hash(d):
    h = hashlib.sha1()
    d = Path(d)
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if p.is_dir() or any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if p.stat().st_size > 4_000_000:
            continue
        h.update(str(rel).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


MULTI = re.compile(
    r"(#\s*include\s*<(thread|future|execution|pthread\.h|omp\.h|sched\.h|sys/wait\.h|spawn\.h)>|std::j?thread(?!::current\b)|std::async|"
    r"pthread_create|#\s*pragma\s+omp|\bv?fork\s*\(|sched_setaffinity|pthread_setaffinity|\bsystem\s*\(|\bpopen\s*\(|"
    r"\bexec[lv][pe]*\s*\(|posix_spawn|\brayon\b|std::process\b|\bthread::(spawn|scope|Builder)\b|\btokio\b|\bcrossbeam\b|"
    r"\blibc::fork\b|\bmultiprocessing\b|"
    r"concurrent\.futures|os\.fork|\bsubprocess\b|\bthreading\b)")


def single_core_violation(d, skip=None, allow=None):
    """One instance, one core: a solver source tree may not contain threads, child processes or affinity changes.
    `skip`: relative paths of files that the problem plugin keeps byte-identical to the original (its LOCKED files).
    `allow`: a regular expression of constructs the plugin tolerates because it disables them at run time (a host whose
    algorithm files carry OpenMP directives, run with a thread limit of one); they are removed from a line before the
    scan, everything else on the line is still checked."""
    skip = set(skip or ())
    allow = re.compile(allow) if isinstance(allow, str) else allow
    for p in sorted(Path(d).rglob("*")):
        rel = p.relative_to(d)
        if rel.as_posix() in skip or any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.is_file() and p.suffix.lower() in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs", ".py", ".java", ".go"):
            for i, line in enumerate(p.read_text(errors="ignore").split("\n"), 1):
                m = MULTI.search(allow.sub(" ", line) if allow is not None else line)
                if m:
                    return ("BUILD REFUSED: the solver must run as ONE thread in ONE process on ONE core. Found `%s` in %s line %d. "
                            "Threads, parallel libraries, child processes and CPU-affinity changes are not allowed." % (
                                m.group(0), p.relative_to(d), i))
    return None


def copy_tree(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP_DIRS, ".*"), symlinks=False)


def fmt(x, nd=4):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def reldiff(x, base):
    """100 * (x - base) / base, the paired difference in per cent of the baseline's cost. A baseline cost of 0 (a
    cost that cannot be lowered, e.g. a solution without any penalty) has no relative difference: 0 when x is 0
    too, +100 when x is worse (bounded: worse by the whole of its own cost). Added 2026-11-19 after the judge
    divided by zero on a development instance that the committed version solved to cost 0."""
    if base:
        return 100.0 * (x - base) / base
    return 0.0 if not x else 100.0


class Judge:
    def __init__(self, camp_dir):
        self.cd = Path(camp_dir).resolve()
        self.C = json.loads((self.cd / "camp.json").read_text())
        self.P = importlib.import_module("problems." + self.C["problem"])
        self.priv = self.cd / "priv"
        self.W = self.cd / "enh" / "work"
        self.G = self.cd / "gen" / "work"
        self.pool = self.C["pool"]
        for sub in ("builds", "cache", "evals", "versions", "fresh", "gen"):
            (self.priv / sub).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.eval_lock = threading.Lock()      # an evaluation / commit and a training-set refresh never overlap
        sp = self.priv / "state.json"
        self.st = json.loads(sp.read_text()) if sp.exists() else {
            "version": 0, "hash": None, "evals": 0, "batches": [], "active": [], "gen_version": 0, "last_eval": None,
            "eps_train": self.C.get("eps_train", 0.0), "eps_fresh": self.C.get("eps_fresh", 0.0), "t0": None}
        self.bks = json.loads(Path(self.C["bks"]).read_text())
        ap = self.priv / "train" / "active.json"
        if ap.exists() and "train_order" not in self.st:
            self.st["train_order"] = json.loads(ap.read_text())      # oldest first
        self.load_train()
        rp = self.priv / "fresh" / "ref.json"
        self.fref = json.loads(rp.read_text()) if rp.exists() else {}
        self.toolbin = self.C.get("toolchain_bin") or os.environ.get("SE_TOOLCHAIN_BIN", "/opt/alphaco/toolchain/bin")

    def load_train(self):
        """(name, path, size) of the training instances; with a rotating training set, only the active ones."""
        act = self.st.get("train_order")
        tr = []
        for p in sorted(Path(self.C["train_dir"]).glob(self.P.INSTANCE_GLOB)):
            if p.stem in self.bks and (act is None or p.stem in act):
                tr.append((p.stem, str(p), self.P.size(str(p))))
        self.train = sorted(tr, key=lambda t: (t[2], t[0]))

    def band(self, n):
        """Size band of an instance: quartiles of the training benchmark's sizes."""
        ns = sorted(t[2] for t in self.train)
        q = [ns[len(ns) // 4], ns[len(ns) // 2], ns[3 * len(ns) // 4]]
        if n < q[0]:
            return "1: n<%d" % q[0]
        if n < q[1]:
            return "2: %d<=n<%d" % (q[0], q[1])
        if n < q[2]:
            return "3: %d<=n<%d" % (q[1], q[2])
        return "4: n>=%d" % q[2]

    def tl(self, n):
        return self.P.time_limit(n) * float(self.C.get("time_scale", 1.0))

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self):
        with self.lock:
            tmp = self.priv / "state.json.tmp"
            tmp.write_text(json.dumps(self.st, indent=1))
            os.replace(tmp, self.priv / "state.json")
            tmp = self.priv / "fresh" / "ref.json.tmp"
            tmp.write_text(json.dumps(self.fref))
            os.replace(tmp, self.priv / "fresh" / "ref.json")
            if self.C.get("dynamic_ref"):
                tmp = Path(self.C["bks"] + ".tmp")
                tmp.write_text(json.dumps(self.bks))
                os.replace(tmp, self.C["bks"])
                try:
                    (self.W / "targets.json").write_text(json.dumps({t[0]: self.bks.get(t[0]) for t in self.train}, indent=0))
                except OSError:
                    pass

    def ledger(self, rec):
        rec["t"] = time.time()
        rec["since_t0"] = (rec["t"] - self.st["t0"]) if self.st.get("t0") else None
        with self.lock, open(self.priv / "ledger.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def cache(self, h):
        p = self.priv / "cache" / (h + ".json")
        return json.loads(p.read_text()) if p.exists() else {}

    def cache_put(self, h, recs):
        with self.lock:
            c = self.cache(h)
            c.update({r["key"]: r for r in recs})
            tmp = self.priv / "cache" / (h + ".json.tmp")
            tmp.write_text(json.dumps(c))
            os.replace(tmp, self.priv / "cache" / (h + ".json"))

    # ── build / run ──────────────────────────────────────────────────────────
    def build(self, tree):
        h = tree_hash(tree)
        b = self.priv / "builds" / h
        with self.lock:
            if (b / "OK").exists():
                return h, str(b / "bin"), None
            if (b / "FAIL").exists():
                return h, None, (b / "FAIL").read_text()
            copy_tree(tree, b / "src")
            (b / "bin").mkdir(parents=True, exist_ok=True)
            bad = (self.P.locked_violation(b / "src") if hasattr(self.P, "locked_violation") else None) or \
                single_core_violation(b / "src", getattr(self.P, "LOCKED", None), getattr(self.P, "SCAN_ALLOW", None))
            if bad:
                (b / "FAIL").write_text(bad)
                return h, None, bad
        env = dict(os.environ)
        env["PATH"] = self.toolbin + ":" + env.get("PATH", "")
        try:
            r = subprocess.run(["bash", "-c", self.P.BUILD, "build", str(b / "bin")], cwd=str(b / "src"), env=env,
                               capture_output=True, text=True, timeout=900)
            out, rc = (r.stdout + r.stderr)[-6000:], r.returncode
        except subprocess.TimeoutExpired:
            out, rc = "build timed out", 1
        if rc != 0:
            (b / "FAIL").write_text(out)
            return h, None, out
        (b / "OK").write_text(out)
        return h, str(b / "bin"), None

    def items(self, suite, seeds=None):
        """[(key, name, path, T, seed, opts, group)]  group: 'train' or 'fresh'."""
        out = []
        seeds = seeds or self.C.get("seeds", [1])
        tr = self.train if suite in ("full", "train") else self.train[2::5]
        opts = self.C.get("train_opts", {})
        for name, path, n in tr:
            for s in seeds:
                out.append(("train__%s__s%d" % (name, s), name, path, self.tl(n), s, opts, "train"))
        if self.C.get("fresh") and suite != "train":
            for b in self.st["active"]:
                fl = sorted((self.priv / "fresh" / b / "inst").glob(self.P.INSTANCE_GLOB))
                if suite == "dev":
                    fl = fl[::2]
                for p in fl:
                    for s in seeds:
                        out.append(("fresh__%s__s%d" % (p.stem, s), p.stem, str(p), self.tl(self.P.size(str(p))),
                                    s, opts, "fresh"))
        return out

    def run_items(self, plan, tag, prio=5):
        """plan: [(hash, bindir, item)]; runs everything in one pool submission; returns {(hash, key): rec}."""
        od = self.priv / "evals" / tag
        od.mkdir(parents=True, exist_ok=True)
        jobs, meta = [], []
        for h, bindir, it in plan:
            key, name, path, T, seed, opts, grp = it
            sol = od / ("%s__%s.sol" % (h[:8], key))
            log = od / ("%s__%s.log" % (h[:8], key))
            jobs.append({"cmd": self.P.run_cmd(bindir, path, str(sol), T, seed, opts), "cwd": str(od),
                         "timeout": T + 120 + self.io_slack(path), "stdout": str(log)})
            meta.append((h, it, sol, log))
        res = poolclient.run_jobs(self.pool, jobs, prio=prio, tag=self.C.get("name")) if jobs else []
        out, by_h = {}, {}
        for (h, it, sol, log), r in zip(meta, res):
            key, name, path, T, seed, opts, grp = it
            v = self.P.verify(path, str(sol), opts)
            slack = 5.0 + 0.03 * T + self.io_slack(path)
            ok = bool(v["ok"]) and not r["timed_out"] and r["elapsed"] <= T + slack
            msg = v.get("msg", "")
            if v["ok"] and not ok:
                msg = "time limit exceeded (%.1fs > %.1fs)" % (r["elapsed"], T)
            cpu = r.get("cpu_s")
            if ok and cpu is not None and cpu > 1.03 * r["elapsed"] + 2.0:      # one instance, one core
                ok, msg = False, "used more than one core (CPU time %.1fs in %.1fs of wall clock)" % (cpu, r["elapsed"])
            rec = {"key": key, "name": name, "grp": grp, "seed": seed, "T": T, "elapsed": round(r["elapsed"], 2), "cpu_s": cpu,
                   "rc": r["rc"], "ok": ok, "cost": v["cost"] if ok else None, "msg": msg, "log": str(log), "sol": str(sol)}
            out[(h, key)] = rec
            by_h.setdefault(h, []).append(rec)
            if ok and grp == "train" and self.C.get("dynamic_ref"):
                with self.lock:          # no best-known values exist: the reference is the best cost any version has reached
                    if self.bks.get(name) is None or rec["cost"] < self.bks[name]:
                        self.bks[name] = rec["cost"]
            if ok and grp == "fresh":
                with self.lock:
                    if name not in self.fref or rec["cost"] < self.fref[name]:
                        self.fref[name] = rec["cost"]
        for h, recs in by_h.items():
            self.cache_put(h, recs)
        self.save()
        return out

    def io_slack(self, path):
        """Seconds a run may take on top of the time limit for reading the instance (plugin hook io_slack(path); 0 for
        the problems whose instances are small files). The time limit itself is the solver's search time."""
        f = getattr(self.P, "io_slack", None)
        try:
            return float(f(path)) if f else 0.0
        except Exception:
            return 0.0

    def copy_log(self, src, dst):
        """Copy a solver log into an agent's workspace with the service's own directory names removed."""
        try:
            t = open(src, "rb").read().decode("utf-8", "ignore")
            for d in (str(Path(self.C["train_dir"])) + "/", str(self.priv / "fresh") + "/", str(self.priv / "evals") + "/", str(self.priv)):
                t = t.replace(d, "")
            Path(dst).write_text(t)
        except OSError:
            pass

    def ref(self, grp, name):
        return self.bks.get(name) if grp == "train" else self.fref.get(name)

    def metrics(self, its, recs):
        """recs: {key: rec}. Mean gap (%) per group, hits on train, failures."""
        m = {"train": [], "fresh": []}
        hits = fails = better = 0
        for key, name, path, T, seed, opts, grp in its:
            r = recs.get(key)
            if r is None:
                continue
            if not r["ok"]:
                fails += 1
                continue
            ref = self.ref(grp, name)
            if not ref:
                continue
            g = 100.0 * (r["cost"] - ref) / ref
            m[grp].append(g)
            if grp == "train":
                hits += r["cost"] <= ref + 1e-9
                better += r["cost"] < ref - 1e-9
        mean = lambda v: (sum(v) / len(v)) if v else None   # noqa: E731
        return {"train_gap": mean(m["train"]), "fresh_gap": mean(m["fresh"]), "hits": hits, "n_train": len(m["train"]),
                "n_fresh": len(m["fresh"]), "fails": fails, "below_bks": better}

    def committed_build(self):
        return self.build(self.priv / "versions" / ("v%d" % self.st["version"]) / "src")

    def paired(self, its, cc, kc):
        """Paired comparison of two trees on the training instances: per instance, the relative cost difference
        (first tree - second tree, % of the second) averaged over the seeds measured on BOTH; then mean, standard
        error over the instances and t = mean / SE. Independent of the reference values."""
        by, seeds = {}, set()
        for key, name, path, T, seed, opts, grp in its:
            if grp != "train":
                continue
            r, k = cc.get(key), kc.get(key)
            if r and k and r["ok"] and k["ok"] and k["cost"]:
                by.setdefault(name, []).append(100.0 * (r["cost"] - k["cost"]) / k["cost"])
                seeds.add(seed)
        d = [sum(v) / len(v) for v in by.values()]
        n = len(d)
        if n < 8:
            return None
        m = sum(d) / n
        se = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 / n ** 0.5
        return {"n": n, "seeds": sorted(seeds), "mean": m, "se": se, "t": (m / se) if se > 0 else 0.0,
                "wins": sum(x < -1e-12 for x in d), "losses": sum(x > 1e-12 for x in d)}

    def missing_runs(self, its, trees):
        """trees: [(hash, bindir)]; the runs of `its` not yet in a tree's cache (every tree once), longest first."""
        plan, seen = [], set()
        for hh, bb in trees:
            if hh in seen or bb is None:
                continue
            seen.add(hh)
            c = self.cache(hh)
            plan += [(hh, bb, it) for it in its if it[0] not in c]
        plan.sort(key=lambda t: -t[2][3])   # longest runs first: minimises the makespan on the pinned pool
        return plan

    # ── enhancer requests ────────────────────────────────────────────────────
    def do_eval(self, suite, note):
        if suite not in ("dev", "full"):
            return 2, "usage: ctl.py eval dev|full \"note\""
        if not poolclient.pool_alive(self.pool):
            return 3, "the compute pool is not reachable right now; try again in a few minutes"
        h, bindir, err = self.build(self.W / "src")
        if err:
            return 1, "BUILD FAILED\n" + err
        ch, cbin, cerr = self.committed_build()
        trees = [(h, bindir), (ch, cbin)]
        rt = self.C.get("retest") if suite == "full" else None
        if rt:
            trees.append(self.build(self.priv / "versions" / "v0" / "src")[:2])   # the original, on the same seeds
        seeds = list(self.C.get("seeds", [1]))
        with self.lock:
            self.st["evals"] += 1
            eid = self.st["evals"]
        t0 = time.time()
        nruns = 0
        while True:
            its = self.items(suite, seeds)
            plan = self.missing_runs(its, trees)
            nruns += len(plan)
            self.run_items(plan, "eval_%03d" % eid)
            cc, kc = self.cache(h), self.cache(ch)
            ps = self.paired(its, cc, kc) if ch != h else None
            # adaptive re-test: a tree that is neither clearly better nor clearly not better than the committed version
            # after the seeds measured so far gets one more seed (both trees, same instances), up to max_seeds
            if not rt or ps is None or len(seeds) >= int(rt.get("max_seeds", 3)):
                break
            if any(not r["ok"] for r in (cc.get(it[0]) for it in its) if r):
                break
            if not (-float(rt.get("t_accept", 2.33)) < ps["t"] <= -float(rt.get("t_try", 1.0))):
                break
            seeds.append(max(seeds) + 1)
        mc, mk = self.metrics(its, cc), self.metrics(its, kc)
        txt = self.summary(eid, suite, h, ch, its, cc, kc, mc, mk, time.time() - t0, ps)
        rd = self.W / "runs" / ("eval_%03d" % eid)
        (rd / "logs").mkdir(parents=True, exist_ok=True)
        (rd / "summary.md").write_text(txt)
        rows = ["instance\tgroup\tseed\ttime_limit\telapsed\tcost\treference\tgap_pct\tcommitted_cost\tstatus"]
        for key, name, path, T, seed, opts, grp in its:
            r, k = cc.get(key), kc.get(key)
            ref = self.ref(grp, name)
            g = ("%.4f" % (100.0 * (r["cost"] - ref) / ref)) if (r and r["ok"] and ref) else ""
            rows.append("\t".join(str(x) for x in (name, grp, seed, "%.1f" % T, r["elapsed"] if r else "", r["cost"] if r else "",
                                                   ref, g, k["cost"] if k else "", ("ok" if r["ok"] else r["msg"]) if r else "missing")))
            if r and os.path.exists(r["log"]):
                self.copy_log(r["log"], rd / "logs" / ("%s_s%d.log" % (name, seed)))
        (rd / "per_instance.tsv").write_text("\n".join(rows) + "\n")
        with self.lock:
            self.st["last_eval"] = {"id": eid, "suite": suite, "hash": h, "m": mc, "mk": mk, "committed_hash": ch,
                                    "seeds": seeds, "paired": ps}
        self.save()
        # what the fresh instances change: the verdict of the full gate next to the verdict a benchmark-only protocol
        # would have given for the same tree (the tree itself is kept in priv/builds/<hash>/src for later re-testing)
        m0 = self.orig_metrics(its)
        g_full = self.gate(mc, mk, h, ch, m0, ps=ps)
        g_train = self.gate(mc, mk, h, ch, None, use_fresh=False, ps=ps)
        self.ledger({"type": "eval", "id": eid, "suite": suite, "hash": h, "committed": self.st["version"], "m": mc,
                     "mk": mk, "m0": m0, "note": note, "runs": nruns, "wall": time.time() - t0, "seeds": seeds, "paired": ps,
                     "gate": g_full[0], "gate_why": g_full[1], "gate_benchmark_only": g_train[0]})
        return 0, txt

    def summary(self, eid, suite, h, ch, its, cc, kc, mc, mk, wall, ps=None):
        L = ["# eval_%03d  suite=%s  tree=%s  committed=v%d  (%d runs, %.0fs)" % (eid, suite, h, self.st["version"], len(its), wall), ""]
        L.append("| | this tree | committed v%d |" % self.st["version"])
        L.append("|---|---|---|")
        lab = ("training set: mean gap to the reference", "training set: instances at the reference") if self.C.get("dynamic_ref") else (
            "training benchmark: mean gap to best-known", "training benchmark: instances at best-known")
        L.append("| %s (%%) | %s | %s |" % (lab[0], fmt(mc["train_gap"]), fmt(mk["train_gap"])))
        L.append("| %s | %d / %d | %d / %d |" % (lab[1], mc["hits"], mc["n_train"], mk["hits"], mk["n_train"]))
        if mc["n_fresh"] or mk["n_fresh"]:
            L.append("| fresh instances: mean gap to best cost seen (%%) | %s | %s |" % (fmt(mc["fresh_gap"]), fmt(mk["fresh_gap"])))
        L.append("| failed runs (infeasible / over time / crash) | %d | %d |" % (mc["fails"], mk["fails"]))
        if mc["below_bks"]:
            L.append("")
            L.append("%d training run(s) BELOW the reference value%s" % (mc["below_bks"], " (the reference is updated)." if self.C.get("dynamic_ref") else ": check the solution file, this is either a new record or a bug."))
        bands = {}
        for key, name, path, T, seed, opts, grp in its:
            r, k = cc.get(key), kc.get(key)
            ref = self.ref(grp, name)
            if not (r and r["ok"] and ref):
                continue
            n = self.P.size(path)
            b = (grp, self.band(n))
            e = bands.setdefault(b, [[], []])
            e[0].append(100.0 * (r["cost"] - ref) / ref)
            if k and k["ok"]:
                e[1].append(100.0 * (k["cost"] - ref) / ref)
        L += ["", "| group | size band | runs | this tree gap % | committed gap % |", "|---|---|---|---|---|"]
        for b in sorted(bands):
            e = bands[b]
            L.append("| %s | %s | %d | %.4f | %s |" % (b[0], b[1], len(e[0]), sum(e[0]) / len(e[0]),
                                                      ("%.4f" % (sum(e[1]) / len(e[1]))) if e[1] else "n/a"))
        bad = [r for r in (cc.get(it[0]) for it in its) if r and not r["ok"]]
        if bad:
            L += ["", "Failed runs:"] + ["- %s seed %d: %s" % (r["name"], r["seed"], r["msg"] or "rc=%s" % r["rc"]) for r in bad[:20]]
        L += ["", "Per-instance table: runs/eval_%03d/per_instance.tsv ; solver logs: runs/eval_%03d/logs/" % (eid, eid)]
        m0 = self.orig_metrics(its)
        L += ["", "Original solver on the same instances: training mean gap %s%%, at the reference %d / %d%s." % (
            fmt(m0["train_gap"]), m0["hits"], m0["n_train"],
            (", fresh mean gap %s%%" % fmt(m0["fresh_gap"])) if m0["n_fresh"] else "")]
        if self.in_debt(mk, m0):
            L += ["WARNING: on the current fresh instances the COMMITTED version is worse than the original solver "
                  "(%s%% vs %s%%). An accepted change did not carry over to unseen instances; the next commit must repair this "
                  "(be no worse than the original on the fresh instances) and may keep the training result unchanged." % (
                      fmt(mk["fresh_gap"]), fmt(m0["fresh_gap"]))]
        if ps:
            L += ["", "Paired comparison with the committed version (same instances, same seeds %s; %d instances): mean cost "
                  "difference %+.4f%% (negative = this tree is better), standard error %.4f, t = %+.2f; better on %d instances, "
                  "worse on %d." % (ps["seeds"], ps["n"], ps["mean"], ps["se"], ps["t"], ps["wins"], ps["losses"])]
            if self.C.get("retest") and len(ps["seeds"]) > len(self.C.get("seeds", [1])):
                L += ["After the first measurement this tree was close to the acceptance threshold, so the service measured it "
                      "with more seeds automatically; the figures above and the gate use all of them."]
        if suite == "full":
            ok, why = self.gate(mc, mk, h, ch, m0, ps=ps)
            L += ["", "Commit gate: %s%s" % ("PASS" if ok else "not passed", "" if ok else " (" + why + ")")]
        return "\n".join(L) + "\n"

    def orig_metrics(self, its):
        """The ORIGINAL solver (v0) on the same items (its results are cached: calibration + every batch delivery)."""
        return self.metrics(its, self.cache(tree_hash(self.priv / "versions" / "v0" / "src")))

    def in_debt(self, mk, m0):
        """True when the COMMITTED version is worse than the original on the fresh instances in the gate: this happens
        when a newly delivered batch exposes that an earlier accepted change does not carry over (over-fitting)."""
        return bool(self.C.get("fresh") and m0 and mk["n_fresh"] and mk["fresh_gap"] is not None and m0["fresh_gap"] is not None
                    and mk["fresh_gap"] > m0["fresh_gap"] + self.st["eps_fresh"])

    def gate(self, mc, mk, h, ch, m0=None, use_fresh=True, ps=None):
        """use_fresh=False evaluates the rule a benchmark-only protocol would apply (recorded for the analysis of
        what the fresh instances change; never used to accept a version when the campaign has fresh instances).
        ps: paired comparison with the committed version; with the `retest` option it replaces the mean-gap threshold:
        the mean paired cost difference must be negative by at least t_accept standard errors."""
        if not use_fresh:
            m0 = None
        if h == ch:
            return False, "this tree is identical to the committed version"
        if mc["fails"]:
            return False, "%d failed runs" % mc["fails"]
        if mc["train_gap"] is None or mk["train_gap"] is None:
            return False, "no measurement"
        ef, et = self.st["eps_fresh"], self.st["eps_train"]
        fresh = bool(use_fresh and self.C.get("fresh") and mc["n_fresh"])
        if self.C.get("dynamic_ref") and m0 and m0["train_gap"] is not None and mc["train_gap"] > m0["train_gap"] + et:
            return False, ("on the current training set this tree is worse than the ORIGINAL solver (mean gap %.4f%% vs %.4f%%)" % (
                mc["train_gap"], m0["train_gap"]))
        best_hits = max(self.st.get("best_hits", 0), mk["hits"])
        if not self.C.get("dynamic_ref") and mc["hits"] < best_hits - int(self.C.get("hits_tolerance", 1)):
            return False, ("the number of benchmark instances at their best-known value dropped to %d (the best committed version "
                           "reached %d; a drop of more than %d is not accepted): the goal is to reach the best-known value on "
                           "more instances, not fewer" % (mc["hits"], best_hits, int(self.C.get("hits_tolerance", 1))))
        if fresh and m0 and m0["fresh_gap"] is not None and mc["fresh_gap"] > m0["fresh_gap"] + ef:
            return False, ("on the fresh instances this tree is worse than the ORIGINAL solver (mean gap %.4f%% vs %.4f%%, "
                           "tolerance %.4f pp): a version that loses to the original on unseen instances is not accepted" % (
                               mc["fresh_gap"], m0["fresh_gap"], ef))
        if self.in_debt(mk, m0):
            # repair commit: the committed version loses to the original on the current fresh instances and this tree
            # does not; it is accepted as long as it does not give back the training-benchmark result
            if mc["train_gap"] > mk["train_gap"] + et:
                return False, ("this tree repairs the fresh instances but its training-set mean gap is worse than the "
                               "committed version's by more than the noise (%.4f pp)" % et)
            return True, ""
        rt = self.C.get("retest")
        if rt:
            ta = float(rt.get("t_accept", 2.33))
            if ps is None:
                return False, "no paired measurement against the committed version"
            if not (ps["mean"] < 0 and ps["t"] <= -ta):
                return False, ("the paired comparison with the committed version must show an improvement of at least %.2f "
                               "standard errors (mean cost difference %+.4f%%, standard error %.4f, t = %+.2f, seeds %s)" % (
                                   ta, ps["mean"], ps["se"], ps["t"], ps["seeds"]))
        elif not (mc["train_gap"] < mk["train_gap"] - et):
            return False, "training-set mean gap must improve on the committed version by more than the run-to-run noise (%.4f pp)" % et
        if fresh and mk["fresh_gap"] is not None and mc["fresh_gap"] > mk["fresh_gap"] + ef:
            return False, "mean gap on the fresh instances got worse than the committed version by more than the noise (%.4f pp)" % ef
        return True, ""

    def refuse(self, why, extra=None):
        """A refused commit must be impossible to overlook: answer, NOTICES.md, status and ledger all carry it."""
        with self.lock:
            self.st["last_commit_attempt"] = "REFUSED at %s: %s" % (time.strftime("%H:%M"), why)
            with open(self.W / "NOTICES.md", "a") as f:
                f.write("- %s: COMMIT REFUSED (the committed version is still v%d): %s\n" % (time.strftime("%H:%M"), self.st["version"], why))
        self.save()
        rec = {"type": "commit_refused", "why": why}
        rec.update(extra or {})
        self.ledger(rec)
        return 1, "COMMIT REFUSED — nothing was committed, the committed version is still v%d.\nReason: %s" % (self.st["version"], why)

    def do_commit(self, note):
        le = self.st.get("last_eval")
        h = tree_hash(self.W / "src")
        if not le or le["suite"] != "full" or le["hash"] != h:
            return self.refuse("run `ctl.py eval full` on exactly this tree first")
        ch, cbin, _ = self.committed_build()
        if le.get("committed_hash") != ch:
            return self.refuse("the committed version changed since that eval; run `ctl.py eval full` again")
        # fresh batches may have arrived after that eval: measure this tree (and the committed version) on the missing
        # instances now, so that a commit never fails just because a batch was delivered in between
        its = self.items("full", le.get("seeds"))      # the seeds of that eval (more than one after a re-test)
        _, bindir, err = self.build(self.W / "src")
        if err:
            return self.refuse("BUILD FAILED\n" + err)
        trees = [(h, bindir), (ch, cbin)]
        if self.C.get("retest"):
            trees.append(self.build(self.priv / "versions" / "v0" / "src")[:2])
        plan = self.missing_runs(its, trees)
        if plan:
            if not poolclient.pool_alive(self.pool):
                return self.refuse("the compute pool is not reachable right now; try the commit again in a few minutes")
            self.run_items(plan, "commit_%03d" % le["id"], prio=4)
        cc, kc = self.cache(h), self.cache(ch)
        mc, mk = self.metrics(its, cc), self.metrics(its, kc)
        ps = self.paired(its, cc, kc)
        ok, why = self.gate(mc, mk, h, ch, self.orig_metrics(its), ps=ps)
        if not ok:
            return self.refuse(why + (" (measured on the current fresh instances: this tree %s%%, committed %s%%)" % (
                fmt(mc["fresh_gap"]), fmt(mk["fresh_gap"])) if mc["n_fresh"] else ""),
                {"hash": h, "m": mc, "mk": mk, "note": note, "paired": ps})
        with self.lock:
            self.st["version"] += 1
            v = self.st["version"]
            copy_tree(self.W / "src", self.priv / "versions" / ("v%d" % v) / "src")
            copy_tree(self.W / "src", self.W / "committed")
            self.st["hash"] = h
            self.st["best_hits"] = max(self.st.get("best_hits", 0), mk["hits"], mc["hits"])
            self.st["last_commit_attempt"] = "accepted at %s as v%d" % (time.strftime("%H:%M"), v)
        self.save()
        self.ledger({"type": "commit", "version": v, "hash": h, "m": mc, "mk": mk, "note": note, "paired": ps,
                     "seeds": le.get("seeds")})
        return 0, "committed as v%d: training mean gap %s%% (was %s%%), at best-known %d/%d (was %d)%s" % (
            v, fmt(mc["train_gap"]), fmt(mk["train_gap"]), mc["hits"], mc["n_train"], mk["hits"],
            (", fresh mean gap %s%% (was %s%%)" % (fmt(mc["fresh_gap"]), fmt(mk["fresh_gap"]))) if mc["n_fresh"] else "")

    def do_revert(self):
        copy_tree(self.priv / "versions" / ("v%d" % self.st["version"]) / "src", self.W / "src")
        return 0, "work tree restored from committed v%d" % self.st["version"]

    # ── protocol v4 (camp option "v4") ───────────────────────────────────────
    # Why (CVRP campaigns of 11-10): (1) the committed version was compared through the very runs that had won its
    # commit, a sample selected for being good, so every later candidate faced an inflated bar; (2) the only instrument
    # was a full single-seed evaluation, ~14 per 4 h on half a pool, with a resolution close to the effects sought;
    # (3) the threshold made agents bundle unverified changes. v4: the committed version's baseline is a set of FRESH
    # runs on its own seeds (measured after the commit, never the winning runs); a cheap screening subset for
    # exploring; `eval full` measures the screening subset first and stops when the tree is clearly not better;
    # the gate is a t-test on per-instance cost differences with automatic re-measurement on more seeds.
    @property
    def paired(self):
        return bool(self.C.get("v4", {}).get("paired"))

    def bseeds(self):
        """Baseline seeds of the committed version. Protocol v5 (v4.paired): its seed block - block 0 = the configured
        base seeds (the calibration of the original), block k = block_step*k+1.. ; the block advances at every commit."""
        V = self.C["v4"]
        bs = list(V["base_seeds"])
        k = int(self.st.get("block", 0)) if self.paired else 0
        # seed_offset (islands, 11-15): every island of a group draws its blocks from its own range, so a tree that one
        # island measured is never measured by another island on the same seeds (the main island's check of an offer
        # taken as it is was a replay of the scout's runs whenever the two were on the same block after a take-over)
        off = int(V.get("seed_offset", 0)) if self.paired else 0
        if k == 0:
            return bs if off == 0 else [off + i + 1 for i in range(len(bs))]
        return [int(V.get("block_step", 1000)) * k + off + i + 1 for i in range(len(bs))]

    def kseeds(self, seeds, bs):
        """Seeds of the committed version to compare a candidate measured on `seeds` with: the same seeds when paired."""
        return list(seeds) if self.paired else list(bs)

    def v4_items(self, names, seeds):
        opts = self.C.get("train_opts", {})
        tr = {t[0]: t for t in self.train}
        return [("train__%s__s%d" % (nm, s), nm, tr[nm][1], self.tl(tr[nm][2]), s, opts, "train")
                for nm in names if nm in tr for s in seeds]

    @staticmethod
    def v4_costs(cache, nm, seeds):
        out = []
        for s in seeds:
            r = cache.get("train__%s__s%d" % (nm, s))
            if r and r["ok"]:
                out.append(r["cost"])
        return out

    def v4_diff(self, names, cc, cseeds, kc, kseeds):
        """Tree cc against tree kc: per instance 100 * (mean cost over cseeds - mean cost over kseeds) / the latter;
        mean over instances, standard error, t, wins/losses, and the mean per size band. Negative = cc is better."""
        size = {t[0]: t[2] for t in self.train}
        d = []
        for nm in names:
            if nm not in size:
                continue
            a, b = self.v4_costs(cc, nm, cseeds), self.v4_costs(kc, nm, kseeds)
            if a and b:
                mb = sum(b) / len(b)
                d.append((nm, reldiff(sum(a) / len(a), mb)))
        n = len(d)
        if n < int(self.C.get("v4", {}).get("min_inst", 8)):
            return None
        x = [v for _, v in d]
        m = sum(x) / n
        se = (sum((v - m) ** 2 for v in x) / (n - 1)) ** 0.5 / n ** 0.5
        bands = {}
        for nm, v in d:
            bands.setdefault(self.band(size[nm]), []).append(v)
        return {"n": n, "seeds": list(cseeds), "base_seeds": list(kseeds), "mean": m, "se": se, "t": (m / se) if se > 0 else 0.0,
                "wins": sum(v < -1e-12 for v in x), "losses": sum(v > 1e-12 for v in x),
                "bands": {b: [len(v), sum(v) / len(v)] for b, v in sorted(bands.items())}}

    def first_round(self, bs=None):
        """Seeds of the first round of a complete evaluation (paired protocol: the head of the current block)."""
        bs = self.bseeds() if bs is None else bs
        return bs[:max(1, len(self.C["v4"].get("cand_seeds", [11])))] if self.paired else bs

    def lazy(self):
        """v4.lazy_baseline (11-16; the independent-set run: with seed_offset a secondary island re-measured its
        committed version on all 8 seeds of its own block after every commit and every take-over, a third of its pool
        time, mostly for seeds no evaluation ever used). 0 = as before: every seed of the block, in the background,
        right after a commit or a take-over; 1 = only the first-round seeds then, the others when a second round
        starts; 2 = nothing in the background: an evaluation measures what it needs when it needs it."""
        return int(self.C["v4"].get("lazy_baseline", 0)) if self.paired else 0

    def v4_baseline(self, tag, prio=5, seeds=None):
        """The committed version and the original, measured on the baseline seeds on every current training instance
        (only the missing runs are started). Returns their tree hashes. seeds: the committed version's seeds to
        measure (default: its whole block, or the first-round seeds with v4.lazy_baseline)."""
        if seeds is None:
            seeds = self.first_round() if self.lazy() else self.bseeds()
        its = self.v4_items([t[0] for t in self.train], list(seeds))
        ch, cb, _ = self.committed_build()
        h0, b0, _ = self.build(self.priv / "versions" / "v0" / "src")
        # the original is measured on the initial training set (calibration); on instances that arrive by rotation
        # only with orig_on_new (it doubles the cost of a refresh; the comparison with the original on those
        # instances can be measured after the campaign, it is not needed to decide a commit)
        trees = [(ch, cb)] + ([(h0, b0)] if self.C["v4"].get("orig_on_new") else [])
        plan = self.missing_runs(its, trees)
        if plan:
            self.run_items(plan, tag, prio=prio)
        return ch, h0

    def v4_screen_names(self, ch, h0):
        """Screening subset: among the training instances on which the baseline runs (committed + original) do not all
        return the same cost (the others carry no signal), every k-th by size."""
        bs, bs0 = self.bseeds(), self.C["v4"]["base_seeds"]
        kc, c0 = self.cache(ch), self.cache(h0)
        uns = [nm for nm, _, _ in self.train
               if len(set(self.v4_costs(kc, nm, bs)) | set(self.v4_costs(c0, nm, bs0))) != 1]
        if len(uns) < int(self.C["v4"].get("min_inst", 8)):
            return [t[0] for t in self.train]
        k = max(1, int(self.C["v4"].get("screen_frac", 3)))
        sub = uns[k // 2::k]
        return sub if len(sub) >= 12 else uns

    # ── size bands (v4.2) ────────────────────────────────────────────────────
    @property
    def CAND_UNIVERSE(self):               # every seed a candidate tree may have been measured with
        if self.paired:
            return self.bseeds()
        c0 = min(self.C["v4"].get("cand_seeds", [11]))
        return range(c0, c0 + 64)

    @property
    def BASE_UNIVERSE(self):               # every seed a committed version's baseline may have
        if self.paired:
            return self.bseeds()
        return range(1, min(self.C["v4"].get("cand_seeds", [11])))

    def v4_bandstats(self, names, cc, kc):
        """Per size band: tree cc against tree kc with EVERY seed measured so far (candidate seeds 11.., baseline seeds
        1..): per instance the difference of the mean costs, then mean, standard error and t over the band."""
        size = {t[0]: t[2] for t in self.train}
        by = {}
        for nm in names:
            if nm not in size:
                continue
            a, b = self.v4_costs(cc, nm, self.CAND_UNIVERSE), self.v4_costs(kc, nm, self.BASE_UNIVERSE)
            if a and b:
                mb = sum(b) / len(b)
                by.setdefault(self.band(size[nm]), []).append((reldiff(sum(a) / len(a), mb), len(a), len(b)))
        out = {}
        for b, v in sorted(by.items()):
            x = [d for d, _, _ in v]
            n = len(x)
            m = sum(x) / n
            se = ((sum((d - m) ** 2 for d in x) / (n - 1)) ** 0.5 / n ** 0.5) if n > 1 else 0.0
            out[b] = {"n": n, "mean": m, "se": se, "t": (m / se) if se > 0 else 0.0,
                      "cseeds": min(c for _, c, _ in v), "bseeds": min(k for _, _, k in v)}
        return out

    def band_pass(self, pb, ps):
        V = self.C["v4"]
        tb, th = float(V.get("t_band", 2.0)), float(V.get("t_harm", 1.5))
        if not pb or not V.get("bands", True):
            return False, "no band result"
        good = [b for b, e in pb.items() if e["n"] >= 12 and e["mean"] < 0 and e["t"] <= -tb]
        bad = [b for b, e in pb.items() if e["n"] >= 12 and e["mean"] > 0 and e["t"] >= th]
        if good and not bad and ps is not None and ps["t"] <= 0.5:
            return True, ""
        best = min(pb.items(), key=lambda kv: kv[1]["t"])
        return False, ("one size band must improve by at least %.1f standard errors while no band gets worse by %.1f and the "
                       "whole set is not worse (best band now: %s, mean cost difference %+.4f%%, t = %+.2f%s)" % (
                           tb, th, best[0], best[1]["mean"], best[1]["t"],
                           ("; worse: " + ", ".join(bad)) if bad else ""))

    def do_probe4(self, names, note):
        """`probe <instance> ...`: the agent's own small representative set (4-5 instances), several seeds, compared with
        the committed version on the same instances. Indicative only; its runs are re-used by a later `eval full` -
        unless v4.probe_offset is set (11-16; the independent-set run: variants were compared in probes on the first
        seeds of the block and the best one then passed the gate on those very runs, so the paired steps summed to
        -1.78 % while the direct comparison with the original stood at -1.36 %): probes then run on their own seeds
        (block seeds + probe_offset), the committed version is measured on them for the probed instances only, and a
        complete evaluation never sees a probe's runs."""
        V = self.C["v4"]
        po = int(V.get("probe_offset", 0)) if self.paired else 0
        if not poolclient.pool_alive(self.pool):
            return 3, "the compute pool is not reachable right now; try again in a few minutes"
        size = {t[0]: t[2] for t in self.train}
        miss = [n for n in names if n not in size]
        lo, hi = 2, int(V.get("probe_max", 8))
        if miss or not (lo <= len(names) <= hi) or len(set(names)) != len(names):
            return 2, ("usage: ctl.py probe <instance> <instance> ... \"note\"  (%d to %d different instances of the current "
                       "training set, file names without extension; 4 or 5 well-chosen ones are the intended use)%s" % (
                           lo, hi, ("; unknown: " + ", ".join(miss)) if miss else ""))
        h, bindir, err = self.build(self.W / "src")
        if err:
            return 1, "BUILD FAILED\n" + err
        with self.lock:
            self.st["evals"] += 1
            eid = self.st["evals"]
        t0, tag = time.time(), "eval_%03d" % eid
        if po:                             # own seeds: the block's baseline is not needed for a probe
            ch, cb, _ = self.committed_build()
        else:
            ch, h0 = self.v4_baseline(tag)
            cb = self.committed_build()[1]
        bseeds = list(V["base_seeds"]) + [x for x in V.get("extra_base_seeds", [3] if len(V["base_seeds"]) < 3 else []) if x not in V["base_seeds"]]
        seeds = list(V.get("probe_seeds") or (list(V.get("cand_seeds", [11])) + [11, 12, 13])[:max(3, len(V.get("cand_seeds", [11])))])
        if self.paired:                    # this tree and the committed version on the same seeds of the current block
            seeds = [s + po for s in self.bseeds()[:len(seeds)]]
            bseeds = list(seeds)
        plan = self.missing_runs(self.v4_items(names, bseeds), [(ch, cb)]) + self.missing_runs(self.v4_items(names, seeds), [(h, bindir)])
        plan.sort(key=lambda t: -t[2][3])
        nruns = len(plan)
        if plan:
            self.run_items(plan, tag)
        cc, kc = self.cache(h), self.cache(ch)
        wall = time.time() - t0
        its = self.v4_items(names, seeds)
        mc = self.metrics(its, cc)
        mean = lambda v: (sum(v) / len(v)) if v else None   # noqa: E731
        rows, d = [], []
        for nm in names:
            a, b = self.v4_costs(cc, nm, seeds), self.v4_costs(kc, nm, bseeds)
            ma, mb = mean(a), mean(b)
            dv = (100.0 * (ma - mb) / mb) if (ma is not None and mb) else None
            if dv is not None:
                d.append(dv)
            ref = self.bks.get(nm)
            rows.append((nm, size[nm], a, b, dv, (100.0 * (ma - ref) / ref) if (ma is not None and ref) else None))
        n = len(d)
        m = (sum(d) / n) if n else None
        se = ((sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 / n ** 0.5) if n > 1 else None
        L = ["# eval_%03d  suite=probe  tree=%s  committed=v%d  (%d instances chosen by you, %d new runs, %.0fs)" % (
            eid, h, self.st["version"], len(names), nruns, wall), "",
            "This tree: seeds %s. Committed v%d: %s %s. Costs are listed per seed." % (
                seeds, self.st["version"], "the same seeds" if self.paired else "baseline seeds", "" if self.paired else bseeds), "",
            "| instance | size | this tree | committed | mean cost difference (%) | gap of this tree to the reference (%) |", "|---|---|---|---|---|---|"]
        for nm, sz, a, b, dv, g in rows:
            L.append("| %s | %d | %s | %s | %s | %s |" % (nm, sz, " ".join(str(x) for x in a) or "FAILED", " ".join(str(x) for x in b),
                                                         ("%+.4f" % dv) if dv is not None else "n/a", ("%.4f" % g) if g is not None else "n/a"))
        if h != ch and m is not None:
            L += ["", "Mean cost difference over the %d instances: %+.4f%% (negative = this tree is better)%s; better on %d, worse on %d." % (
                n, m, (", standard error %.4f" % se) if se is not None else "", sum(x < -1e-12 for x in d), sum(x > 1e-12 for x in d)),
                "", "A probe is indicative: a handful of instances cannot decide a commit. Its value is speed. Keep the same "
                "representative set from one probe to the next so that results can be compared, and replace an instance when it "
                "stops telling you anything (every version reaches the same cost on it). When a probe looks good, run "
                "`ctl.py eval full` on exactly this tree: " + (
                    "it measures the tree on other seeds (a probe's runs are never part of a commit decision)." if po
                    else "it re-uses these runs.")]
        elif h == ch:
            L += ["", "This tree is identical to the committed version: the differences show how far two measurements of the same "
                  "code differ on these instances (mean %s%%)." % fmt(m)]
        L += ["", "Failed runs: %d. Solver logs: runs/eval_%03d/logs/" % (mc["fails"], eid)]
        txt = "\n".join(L) + "\n"
        rd = self.W / "runs" / ("eval_%03d" % eid)
        (rd / "logs").mkdir(parents=True, exist_ok=True)
        (rd / "summary.md").write_text(txt)
        for key, name, path, T, seed, opts, grp in its:
            r = cc.get(key)
            if r and os.path.exists(r["log"]):
                self.copy_log(r["log"], rd / "logs" / ("%s_s%d.log" % (name, seed)))
        ps = {"n": n, "seeds": seeds, "base_seeds": bseeds, "mean": m, "se": se or 0.0, "t": (m / se) if (se and m is not None) else 0.0,
              "wins": sum(x < -1e-12 for x in d), "losses": sum(x > 1e-12 for x in d), "bands": {}} if n else None
        with self.lock:
            self.st["last_eval"] = {"id": eid, "suite": "probe", "hash": h, "m": mc, "mk": mc, "committed_hash": ch,
                                    "seeds": seeds, "paired": ps, "complete": False}
        self.save()
        self.ledger({"type": "eval", "id": eid, "suite": "probe", "hash": h, "committed": self.st["version"], "m": mc,
                     "mk": mc, "m0": mc, "note": note, "runs": nruns, "wall": wall, "seeds": seeds, "base_seeds": bseeds,
                     "paired": ps, "paired_orig": None, "n_inst": len(names), "sizes": sorted(size[nm] for nm in names),
                     "stopped_early": None, "gate": False, "gate_why": "probe", "gate_benchmark_only": False})
        return 0, txt

    def baseline_async(self, v):
        """Fresh baseline of a newly committed version, in the background: the commit answers at once, and the next
        evaluation (which needs the baseline) simply waits for this to finish. With v4.lazy_baseline 1 only the
        first-round seeds, with 2 nothing (see lazy)."""
        lz = self.lazy()
        if lz >= 2:
            return

        def work():
            try:
                with self.eval_lock:
                    self.v4_baseline("base_v%d" % v, prio=4)
                    names = [t[0] for t in self.train]
                    bs = self.first_round() if lz else self.bseeds()
                    self.ledger({"type": "baseline", "version": v, "base_seeds": bs, "block": self.st.get("block", 0),
                                 "m": self.metrics(self.v4_items(names, bs), self.cache(self.st["hash"]))})
            except Exception:
                with open(self.cd / "logs" / "judge_errors.log", "a") as ef:
                    ef.write("%s baseline v%d\n%s\n" % (time.strftime("%F %T"), v, traceback.format_exc()))
        threading.Thread(target=work, daemon=True).start()

    def round_means(self, names, cc, seeds, kc, bs):
        """Mean cost difference to the committed version in each round of candidate seeds (a round = cand_step seeds)."""
        k = max(1, int(self.C["v4"].get("cand_step", len(self.C["v4"].get("cand_seeds", [11])))))
        out = []
        for i in range(0, len(seeds), k):
            p = self.v4_diff(names, cc, seeds[i:i + k], kc, self.kseeds(seeds[i:i + k], bs))
            out.append(p["mean"] if p else None)
        return out

    def gate4(self, h, ch, mc, ps, ps0, pb=None, rm=None):
        ta = float(self.C["v4"].get("t_accept", 2.33))
        if h == ch:
            return False, "this tree is identical to the committed version"
        if mc["fails"]:
            return False, "%d failed runs" % mc["fails"]
        if ps is None:
            return False, "no measurement"
        if ps0 and ps0["mean"] > 0 and ps0["t"] > 1.0:
            return False, ("on the current training set this tree is worse than the ORIGINAL solver (mean cost difference "
                           "%+.4f%%, t = %+.2f)" % (ps0["mean"], ps0["t"]))
        if self.C["v4"].get("rule") == "decrease2":
            need = int(self.C["v4"].get("rounds", 2))
            rm = rm or []
            if ps["mean"] < 0 and ps["t"] <= -ta:           # clear enough: no further round needed
                return True, ""
            # second gate: lower in every round AND, over all the seeds together, at least t_second standard errors
            # (11-12: "lower in two rounds" alone let changes with t = -0.58 / -0.48 through; the threshold is 0.6)
            t2 = float(self.C["v4"].get("t_second", 0.0))
            if len(rm) >= need and all(m is not None and m < 0 for m in rm) and ps["mean"] < 0 and ps["t"] <= -t2:
                return True, ""
            return False, ("either an improvement of at least %.1f standard errors (now t = %+.2f), or a mean cost lower than "
                           "the committed version's in each of %d consecutive rounds of seeds (rounds measured: %s)%s" % (
                               ta, ps["t"], need, ", ".join("%+.4f%%" % m for m in rm if m is not None) or "none",
                               (" together with an improvement of at least %.1f standard errors over all those seeds" % t2) if t2 > 0 else ""))
        if ps["mean"] < 0 and ps["t"] <= -ta:
            return True, ""
        okb, whyb = self.band_pass(pb, ps)
        if okb:
            return True, ""
        return False, ("the comparison with the committed version must show an improvement of at least %.2f standard "
                       "errors on the whole set (now: mean cost difference %+.4f%%, standard error %.4f, t = %+.2f, seeds %s)%s" % (
                           ta, ps["mean"], ps["se"], ps["t"], ps["seeds"], ("; or else " + whyb) if pb else ""))

    def do_eval4(self, suite, note):
        V = self.C["v4"]
        suite = "screen" if suite == "dev" else suite
        if suite not in ("screen", "full"):
            return 2, "usage: ctl.py eval screen|full \"note\"   (quick test on your own instances: ctl.py probe <instance> ... \"note\")"
        if not poolclient.pool_alive(self.pool):
            return 3, "the compute pool is not reachable right now; try again in a few minutes"
        h, bindir, err = self.build(self.W / "src")
        if err:
            return 1, "BUILD FAILED\n" + err
        with self.lock:
            self.st["evals"] += 1
            eid = self.st["evals"]
        t0, tag = time.time(), "eval_%03d" % eid
        moved = False
        if suite == "full" and self.paired and V.get("fresh_after_fail") and self.st.get("block_spent"):
            # fresh_after_fail (11-15): a complete evaluation on the current seeds did not pass -> those seeds are used
            # up; the next complete evaluation (this tree or a variant of it) is measured on a new block, so that a
            # variant cannot pass on the draw the committed version happened to have on the old one
            with self.lock:
                self.st["block"] = int(self.st.get("block", 0)) + 1
                self.st["block_spent"] = False
                moved = True
            self.save()
            self.ledger({"type": "block_advance", "block": self.st["block"], "eval": eid,
                         "why": "a complete evaluation on the previous seeds did not pass"})
        ch, h0 = self.v4_baseline(tag)
        bs, seeds = self.bseeds(), list(V.get("cand_seeds", [11]))
        bs0 = list(V["base_seeds"])                    # the original's seeds (its calibration runs)
        if self.paired:                                # same seeds as the committed version's baseline, first round
            seeds = bs[:len(seeds)]
        max_seeds = min(int(V.get("max_seeds", 3)), len(bs)) if self.paired else int(V.get("max_seeds", 3))
        ta, tt, ts = float(V.get("t_accept", 2.33)), float(V.get("t_try", 1.0)), float(V.get("t_stop", 0.5))
        allnames = [t[0] for t in self.train]
        scr = self.v4_screen_names(ch, h0)
        kc, c0 = self.cache(ch), self.cache(h0)
        nruns = [0]

        def measure(names):
            its_ = self.v4_items(names, seeds)
            plan = self.missing_runs(its_, [(h, bindir)])
            nruns[0] += len(plan)
            self.run_items(plan, tag)
            cc_ = self.cache(h)
            return cc_, any((it[0] in cc_ and not cc_[it[0]]["ok"]) for it in its_)

        cc, failed = measure(scr)
        names, stopped = scr, None
        if suite == "full":
            ps_s = self.v4_diff(scr, cc, seeds, kc, self.kseeds(seeds, bs))
            band_hit = self.band_pass(self.v4_bandstats(allnames, cc, kc), {"t": 0.0})[0]   # from earlier measurements of this tree
            if h != ch and failed:
                stopped = "a run failed on the screening subset"
            elif h != ch and ps_s and ps_s["t"] > ts and not band_hit:
                stopped = ("on the screening subset (%d instances) this tree is not better than the committed version: mean "
                           "cost difference %+.4f%%, t = %+.2f" % (ps_s["n"], ps_s["mean"], ps_s["t"]))
            else:
                names = allnames
                while True:
                    cc, failed = measure(allnames)
                    ps = self.v4_diff(allnames, cc, seeds, kc, self.kseeds(seeds, bs))
                    # adaptive re-measurement: promising but not yet conclusive -> one more seed, up to max_seeds
                    if h == ch or failed or ps is None or len(seeds) >= max_seeds:
                        break
                    if V.get("rule") == "decrease2":
                        if ps["mean"] < 0 and ps["t"] <= -ta:
                            break
                        rm_ = self.round_means(allnames, cc, seeds, kc, bs)
                        if len(rm_) >= int(V.get("rounds", 2)) or not all(m is not None and m < 0 for m in rm_):
                            break
                        seeds += [max(seeds) + 1 + q for q in range(int(V.get("cand_step", 1)))]
                        if self.lazy():                # the committed version on the new round's seeds, now that they are needed
                            self.v4_baseline(tag, seeds=self.kseeds(seeds, bs))
                            kc = self.cache(ch)
                        continue
                    if not (-ta < ps["t"] <= -tt):
                        break
                    if self.band_pass(self.v4_bandstats(allnames, cc, kc), ps)[0]:
                        break
                    seeds += [max(seeds) + 1 + q for q in range(int(V.get("cand_step", 1)))]
                    if self.lazy():
                        self.v4_baseline(tag, seeds=self.kseeds(seeds, bs))
                        kc = self.cache(ch)
        size_ = {t[0]: t[2] for t in self.train}
        ps = self.v4_diff(names, cc, seeds, kc, self.kseeds(seeds, bs))
        ps0 = self.v4_diff(names, cc, seeds, c0, bs0) if ch != h0 else ps
        its, bits = self.v4_items(names, seeds), self.v4_items(names, self.kseeds(seeds, bs) if self.lazy() else bs)
        mc, mk, m0 = self.metrics(its, cc), self.metrics(bits, kc), self.metrics(self.v4_items(names, bs0), c0)
        complete = suite == "full" and not stopped
        pb = self.v4_bandstats(names, cc, kc) if h != ch else {}
        rm = self.round_means(names, cc, seeds, kc, bs) if h != ch else []
        ok, why = self.gate4(h, ch, mc, ps, ps0, pb, rm) if complete else (False, "not a complete `eval full`")
        wall = time.time() - t0
        L = ["# eval_%03d  suite=%s  tree=%s  committed=v%d  (%d instances, %d new runs, %.0fs)" % (
            eid, suite, h, self.st["version"], len(names), nruns[0], wall), ""]
        if suite == "screen":
            L += ["Screening subset: %d of the %d training instances (those on which the baseline runs do not all return the "
                  "same cost, every %d-th by size). For exploring; only `eval full` counts for a commit." % (
                      len(scr), len(allnames), max(1, int(V.get("screen_frac", 3)))), ""]
        if stopped:
            L += ["STOPPED EARLY: `eval full` measures the screening subset first, and %s. The remaining instances were not "
                  "run. The figures below are for the screening subset." % stopped, ""]
        if self.paired:
            L += ["This tree and the committed v%d are compared on the SAME seeds %s (the committed version's runs on them were "
                  "measured after its commit; a new set of seeds is used after every commit, so no run that won a commit is "
                  "ever part of a comparison). Per instance the mean cost over these seeds is compared." % (self.st["version"], seeds), ""]
        else:
            L += ["This tree: seeds %s. Committed v%d: its baseline, fresh runs on seeds %s (never the runs that won its commit); "
                  "per instance the mean cost over the seeds is compared." % (seeds, self.st["version"], bs), ""]
        if ps and h != ch:
            L += ["Against the committed version: mean cost difference %+.4f%% (negative = this tree is better), standard error "
                  "%.4f, t = %+.2f; better on %d instances, worse on %d, of %d." % (
                      ps["mean"], ps["se"], ps["t"], ps["wins"], ps["losses"], ps["n"])]
            if ps0 is not ps and ps0:
                L += ["Against the original solver: mean cost difference %+.4f%%, standard error %.4f, t = %+.2f." % (
                    ps0["mean"], ps0["se"], ps0["t"])]
            if V.get("rule") == "decrease2" and rm:
                L += ["Rounds of seeds measured for this tree (mean cost difference to the committed version in each): %s. A tree "
                      "is accepted when the first round already shows an improvement of %.1f standard errors, or when the mean "
                      "cost is lower in %d consecutive rounds%s." % (
                          ", ".join("%+.4f%%" % m for m in rm if m is not None), ta, int(V.get("rounds", 2)),
                          (" and the improvement over all those seeds together is at least %.1f standard errors" % float(V["t_second"]))
                          if float(V.get("t_second", 0)) > 0 else "")]
            L += ["", "| size band | instances | mean cost difference vs committed (%) |", "|---|---|---|"]
            L += ["| %s | %d | %+.4f |" % (b, v[0], v[1]) for b, v in ps["bands"].items()]
            if pb:
                L += ["", "Per size band, with every seed measured for this tree so far%s:" % (
                    "" if int(V.get("probe_offset", 0)) else " (probes of this tree included)"),
                      "", "| size band | instances | seeds: this tree / committed | mean cost difference (%) | standard error | t |",
                      "|---|---|---|---|---|---|"]
                L += ["| %s | %d | %d / %d | %+.4f | %.4f | %+.2f |" % (b, e["n"], e["cseeds"], e["bseeds"], e["mean"], e["se"], e["t"])
                      for b, e in pb.items()]
            L += ["", "How to read t: about -1 to +1 is indistinguishable from run-to-run noise; below -2 the improvement is "
                  "very likely real; above +2 the tree is very likely worse. The standard error tells the smallest effect "
                  "this measurement can see (about twice its value)."]
            if len(seeds) > len(V.get("cand_seeds", [11])):
                L += ["After the first seed this tree was promising but not conclusive, so the service measured it on more "
                      "seeds automatically; all figures use all of them."]
        elif h == ch:
            L += ["This tree is identical to the committed version: the figures show how far two measurements of the same "
                  "code can differ. Mean cost difference to its baseline %s%%." % (fmt(ps["mean"]) if ps else "n/a")]
        gb = {}
        for nm in names:
            ref = self.bks.get(nm)
            if not ref or nm not in size_:
                continue
            e = gb.setdefault(self.band(size_[nm]), [[], [], []])
            for k, (cache_, sd_) in enumerate(((cc, seeds), (kc, self.kseeds(seeds, bs)), (c0, bs0))):
                v = self.v4_costs(cache_, nm, sd_)
                if v:
                    e[k].append(100.0 * (sum(v) / len(v) - ref) / ref)
        mg = lambda v: ("%.4f" % (sum(v) / len(v))) if v else "n/a"   # noqa: E731
        L += ["", "Where the remaining loss is: mean gap (%) to the reference, per size band. The reference of an instance is "
              "the best cost any run has reached on it, including longer runs; a band with a large gap is a band where the "
              "solver, in the time it is given, stays far from what is known to be reachable.",
              "", "| size band | instances | this tree | committed | original |", "|---|---|---|---|---|"]
        L += ["| %s | %d | %s | %s | %s |" % (b, len(e[0]) or len(e[1]), mg(e[0]), mg(e[1]), mg(e[2])) for b, e in sorted(gb.items())]
        L += ["", "Mean gap to the reference (the best cost any run has reached so far; it is not an optimum): this tree %s%%, "
              "committed %s%%, original %s%%. Failed runs of this tree: %d." % (
                  fmt(mc["train_gap"]), fmt(mk["train_gap"]), fmt(m0["train_gap"]), mc["fails"])]
        bad = [r for r in (cc.get(it[0]) for it in its) if r and not r["ok"]]
        if bad:
            L += ["", "Failed runs:"] + ["- %s seed %d: %s" % (r["name"], r["seed"], r["msg"] or "rc=%s" % r["rc"]) for r in bad[:20]]
        if suite == "full":
            L += ["", "Commit gate: %s%s" % ("PASS" if ok else "not passed", "" if ok else " (" + why + ")")]
        spent = bool(complete and h != ch and not ok and self.paired and V.get("fresh_after_fail"))
        if moved:
            L += ["", "This evaluation used a NEW set of seeds %s: the previous complete evaluation on the earlier seeds did not "
                  "pass, and seeds on which a tree has failed are not used for a commit decision again (the committed version "
                  "was measured on the new seeds first)." % bs]
        if spent:
            L += ["", "These seeds are used up now: the next `ctl.py eval full` - of this tree or of a variant - will be measured on "
                  "a new set of seeds. `ctl.py probe` and `ctl.py eval screen` stay on the current seeds until then."]
        L += ["", "Per-instance table: runs/eval_%03d/per_instance.tsv ; solver logs: runs/eval_%03d/logs/" % (eid, eid)]
        txt = "\n".join(L) + "\n"
        rd = self.W / "runs" / ("eval_%03d" % eid)
        (rd / "logs").mkdir(parents=True, exist_ok=True)
        (rd / "summary.md").write_text(txt)
        size = {t[0]: t[2] for t in self.train}
        rows = ["instance\tsize\tthis_tree_mean_cost\tcommitted_mean_cost\tdiff_pct\toriginal_mean_cost\treference\tstatus"]
        mean = lambda v: (sum(v) / len(v)) if v else None   # noqa: E731
        for nm in names:
            a, b, o = mean(self.v4_costs(cc, nm, seeds)), mean(self.v4_costs(kc, nm, self.kseeds(seeds, bs))), mean(self.v4_costs(c0, nm, bs0))
            st_ = "ok" if a is not None else "failed"
            rows.append("\t".join(str(x) for x in (nm, size.get(nm, ""), "" if a is None else round(a, 2), "" if b is None else round(b, 2),
                                                   ("%.4f" % (100.0 * (a - b) / b)) if (a is not None and b) else "",
                                                   "" if o is None else round(o, 2), self.bks.get(nm), st_)))
        (rd / "per_instance.tsv").write_text("\n".join(rows) + "\n")
        for key, name, path, T, seed, opts, grp in its:
            r = cc.get(key)
            if r and os.path.exists(r["log"]):
                self.copy_log(r["log"], rd / "logs" / ("%s_s%d.log" % (name, seed)))
        with self.lock:
            self.st["last_eval"] = {"id": eid, "suite": suite, "hash": h, "m": mc, "mk": mk, "committed_hash": ch,
                                    "seeds": seeds, "paired": ps, "complete": complete, "gate": bool(ok)}
            if spent:
                self.st["block_spent"] = True
        self.save()
        self.ledger({"type": "eval", "id": eid, "suite": suite, "hash": h, "committed": self.st["version"], "m": mc, "mk": mk,
                     "block": self.st.get("block", 0), "block_spent": spent,
                     "m0": m0, "note": note, "runs": nruns[0], "wall": wall, "seeds": seeds, "base_seeds": bs, "paired": ps,
                     "paired_orig": ps0, "n_inst": len(names), "stopped_early": stopped, "gate": ok, "gate_why": why,
                     "gate_benchmark_only": ok, "bands": pb, "rounds": rm})
        return 0, txt

    def do_commit4(self, note):
        V = self.C["v4"]
        le = self.st.get("last_eval")
        h = tree_hash(self.W / "src")
        if not le or le["suite"] != "full" or le["hash"] != h or not le.get("complete"):
            return self.refuse("run `ctl.py eval full` on exactly this tree first (an evaluation that was stopped early does not count)")
        ch, cbin, _ = self.committed_build()
        if le.get("committed_hash") != ch:
            return self.refuse("the committed version changed since that eval; run `ctl.py eval full` again")
        if not poolclient.pool_alive(self.pool):
            return self.refuse("the compute pool is not reachable right now; try the commit again in a few minutes")
        _, bindir, err = self.build(self.W / "src")
        if err:
            return self.refuse("BUILD FAILED\n" + err)
        tag = "commit_%03d" % le["id"]
        names, seeds, bs = [t[0] for t in self.train], list(le["seeds"]), self.bseeds()
        # (the training set may have been refreshed since the eval)
        ch, h0 = self.v4_baseline(tag, prio=4, seeds=(self.kseeds(seeds, bs) if self.lazy() else None))
        bs0 = list(V["base_seeds"])
        its = self.v4_items(names, seeds)
        plan = self.missing_runs(its, [(h, bindir)])
        if plan:
            self.run_items(plan, tag, prio=4)
        cc, kc, c0 = self.cache(h), self.cache(ch), self.cache(h0)
        ps = self.v4_diff(names, cc, seeds, kc, self.kseeds(seeds, bs))
        ps0 = self.v4_diff(names, cc, seeds, c0, bs0) if ch != h0 else ps
        mc, mk = self.metrics(its, cc), self.metrics(self.v4_items(names, self.kseeds(seeds, bs) if self.lazy() else bs), kc)
        pb = self.v4_bandstats(names, cc, kc)
        rm = self.round_means(names, cc, seeds, kc, bs)
        ok, why = self.gate4(h, ch, mc, ps, ps0, pb, rm)
        if not ok:
            return self.refuse(why, {"hash": h, "m": mc, "mk": mk, "note": note, "paired": ps, "bands": pb})
        with self.lock:
            self.st["version"] += 1
            v = self.st["version"]
            copy_tree(self.W / "src", self.priv / "versions" / ("v%d" % v) / "src")
            copy_tree(self.W / "src", self.W / "committed")
            self.st["hash"] = h
            self.st["last_commit_attempt"] = "accepted at %s as v%d" % (time.strftime("%H:%M"), v)
            self.st["claimed"] = False             # islands: the next direction must be declared before the next evaluation
            if self.paired:                        # v5: a new seed block for the new committed version
                self.st["block"] = int(self.st.get("block", 0)) + 1
                self.st["block_spent"] = False
        bs = self.bseeds()
        self.save()
        self.ledger({"type": "commit", "version": v, "hash": h, "m": mc, "mk": mk, "note": note, "paired": ps,
                     "paired_orig": ps0, "seeds": seeds, "bands": pb, "block": self.st.get("block", 0),
                     "rounds": rm,
                     "rule": V.get("rule") or ("whole set" if (ps["mean"] < 0 and ps["t"] <= -float(V.get("t_accept", 2.33))) else "band")})
        if not self.st.get("sync_deferred"):            # (a take-over follows at once: its version gets the baseline)
            self.baseline_async(v)                      # fresh baseline of the new committed version, in the background
        return 0, ("committed as v%d: mean cost difference to the previous committed version %+.4f%% (standard error %.4f, "
                   "t = %+.2f, seeds %s). The new committed version is measured again with fresh runs on other seeds (%s); "
                   "later comparisons use those, not the runs that won this commit (an evaluation waits for the ones it needs)."
                   % (v, ps["mean"], ps["se"], ps["t"], seeds, bs))

    def do_status(self):
        le = self.st.get("last_eval")
        h = tree_hash(self.W / "src")
        L = ["committed version: v%d (tree %s)" % (self.st["version"], self.st["hash"]),
             "last commit attempt: %s" % self.st.get("last_commit_attempt", "none"),
             "work tree: %s (%s)" % (h, "same as committed" if h == self.st["hash"] else "differs from committed"),
             "evaluations so far: %d" % self.st["evals"],
             "training instances: %d ; fresh instances in the commit gate: %d (batches %s)" % (
                 len(self.train), sum(len(list((self.priv / "fresh" / b / "inst").glob(self.P.INSTANCE_GLOB))) for b in self.st["active"]),
                 ", ".join(self.st["active"]) or "none yet")]
        if le:
            L.append("last eval: eval_%03d suite=%s tree=%s train gap %s%% hits %d fresh gap %s%%" % (
                le["id"], le["suite"], le["hash"], fmt(le["m"]["train_gap"]), le["m"]["hits"], fmt(le["m"]["fresh_gap"])))
        return 0, "\n".join(L)

    def do_run(self, names, seed, long=False):
        if not poolclient.pool_alive(self.pool):
            return 3, "the compute pool is not reachable right now; try again in a few minutes"
        allit = {it[1]: it for it in self.items("full")}
        miss = [n for n in names if n not in allit]
        if miss or not names:
            return 2, "unknown instance(s) %s (use file names without extension)" % ", ".join(miss)
        if len(names) > (8 if long else 32):
            return 2, "at most %d instances per call" % (8 if long else 32)
        F = float(self.C.get("long_factor", 5.0)) if long else 1.0
        h, bindir, err = self.build(self.W / "src")
        if err:
            return 1, "BUILD FAILED\n" + err
        stamp = time.time_ns() % 10**9
        plan = []
        for nm in names:
            key, _, path, T, _, opts, grp = allit[nm]
            plan.append((h, bindir, ("single__%s__s%d__%d" % (nm, seed, stamp), nm, path, T * F, seed, opts, grp)))
        res = self.run_items(plan, "single", prio=3)
        (self.W / "runs" / "single").mkdir(parents=True, exist_ok=True)
        L = []
        for _, _, it in plan:
            r = res[(h, it[0])]
            nm, grp = it[1], it[6]
            ref = self.ref(grp, nm)
            dst = self.W / "runs" / "single" / ("%s_s%d_%s.log" % (nm, seed, h[:8]))
            self.copy_log(r["log"], dst)
            g = (" gap %.4f%% (reference %s)" % (100.0 * (r["cost"] - ref) / ref, ref)) if (r["ok"] and ref) else ""
            L.append("%s seed %d%s: %s in %.1fs of %.1fs%s  [log: runs/single/%s]" % (
                nm, seed, (" LONG RUN (%g x the time limit)" % F) if long else "",
                ("cost %s" % r["cost"]) if r["ok"] else ("FAILED: " + (r["msg"] or "rc=%s" % r["rc"])), r["elapsed"], it[3], g, dst.name))
        return 0, "\n".join(L)

    # ── instance agent requests ──────────────────────────────────────────────
    def do_submit(self, note):
        gdir = self.G / "gen"
        if not (gdir / "gen.py").exists() or not (gdir / "families.json").exists():
            return 2, "submit needs gen/gen.py and gen/families.json"
        try:
            fams = [f["name"] for f in json.loads((gdir / "families.json").read_text())]
            assert fams and all(isinstance(x, str) and x.replace("_", "").isalnum() for x in fams)
        except Exception:
            return 2, "gen/families.json must be a list of {\"name\": <alphanumeric id>, \"description\": ...}"
        with self.lock:
            self.st["gen_version"] += 1
            gv = self.st["gen_version"]
            bname = "b%02d" % (len(self.st["batches"]) + 1)
        copy_tree(gdir, self.priv / "gen" / ("v%d" % gv))
        B = int(self.C.get("batch_size", 16))
        rng = random.Random("%s-%d-%s" % (self.C.get("secret", "s"), gv, time.time_ns()))
        accepted, rejected, meta = self.generate(fams, B, rng, bname, self.priv / "fresh" / bname / "inst", "G%02d" % (len(self.st["batches"]) + 1))
        if len(accepted) < max(4, B // 2):
            self.ledger({"type": "batch_rejected", "gen_version": gv, "accepted": len(accepted), "rejected": rejected[:20]})
            shutil.rmtree(self.priv / "fresh" / bname, ignore_errors=True)
            return 1, "batch rejected: only %d of %d generated instances were valid.\n%s" % (
                len(accepted), B, "\n".join("- %s n=%s: %s" % r for r in rejected[:20]))
        return self.finish_batch(bname, accepted, rejected, meta, gv, note)

    def draw_batch(self, note="scheduled draw from the benchmark twin"):
        """Phase 1 of architecture v2: the judge itself draws a fresh batch from the certified benchmark twin."""
        import twin as twinmod
        T = twinmod.Twin(self.C["twin"])
        with self.lock:
            bname = "b%02d" % (len(self.st["batches"]) + 1)
        B = int(self.C.get("batch_size", 16))
        rng = random.Random("%s-draw-%s-%d" % (self.C.get("secret", "s"), bname, time.time_ns()))
        ns = [t[2] for t in self.train]
        sizes = [ns[min(len(ns) - 1, int((i + rng.random()) / B * len(ns)))] for i in range(B)]   # sizes of the public benchmark
        outdir = self.priv / "fresh" / bname / "inst"
        res = T.draw(B, outdir, "G%02d" % (len(self.st["batches"]) + 1), sizes, rng)
        accepted = sorted(outdir.glob(self.P.INSTANCE_GLOB))
        rejected = [(d["family"], d["n"], d["why"]) for d in res.values() if not d["ok"]]
        meta = {"inst": {k: {"family": d["family"], "n": d["n"], "seed": d["seed"]} for k, d in res.items() if d["ok"]},
                "twin_version": T.st["accepted"]}
        with self.lock:
            self.st["last_draw"] = time.time()
        if len(accepted) < max(4, B // 2):
            self.ledger({"type": "batch_rejected", "twin_version": T.st["accepted"], "accepted": len(accepted), "rejected": rejected[:20]})
            shutil.rmtree(self.priv / "fresh" / bname, ignore_errors=True)
            self.save()
            return 1, "draw rejected"
        return self.finish_batch(bname, accepted, rejected, meta, T.st["accepted"], note)

    def batch_loop(self):
        interval = float(self.C.get("batch_interval_s", 1500))
        fixed = self.C.get("fresh_mode") == "fixed"            # ablation arm B: one batch, never renewed
        last_v = self.st["version"]
        while not (self.cd / "STOP").exists():
            live = [b for b in self.st["batches"] if b not in self.st.get("retired", [])]
            since = time.time() - self.st.get("last_draw", 0)
            due = (not live and since > 300) or (live and not fixed and (since >= interval or (self.st["version"] != last_v and since >= 600)))
            if due:
                last_v = self.st["version"]
                try:
                    self.draw_batch()
                except Exception:
                    with self.lock:
                        self.st["last_draw"] = time.time()
                    with open(self.cd / "logs" / "judge_errors.log", "a") as ef:
                        ef.write("%s draw_batch\n%s\n" % (time.strftime("%F %T"), traceback.format_exc()))
            time.sleep(10)

    def rotate(self):
        """Rotating training set: replace the k oldest training instances by k new ones of the same sizes drawn from the
        instance source. The new ones are first measured with the original and the committed version (out-of-sample
        record: nobody had seen them), then they become training instances."""
        import twin as twinmod
        T = twinmod.Twin(self.C["train_twin"])
        k = int(self.C["rotate"].get("k", 25))
        with self.lock:
            order = list(self.st["train_order"])
            rot = self.st.get("rotations", 0) + 1
            self.st["last_rotate"] = time.time()
        old = order[:k]
        size = {name: n for name, _, n in self.train}
        rng = random.Random("%s-rot-%d-%d" % (self.C.get("secret", "s"), rot, time.time_ns()))
        tdir = Path(self.C["train_dir"])
        ext = self.P.INSTANCE_GLOB.replace("*", "")
        res = T.draw(len(old), tdir, "R%02d" % rot, [size[o] for o in old], rng)
        new = [nm for nm, d in res.items() if d["ok"]]
        v0h, v0b, _ = self.build(self.priv / "versions" / "v0" / "src")
        ch, cb, _ = self.committed_build()
        seeds, opts = self.C.get("seeds", [1]), self.C.get("train_opts", {})
        if self.C.get("v4"):
            seeds = list(self.C["v4"]["base_seeds"])       # new instances get the baseline runs of both versions
        with self.lock:
            for nm in new:
                self.bks.setdefault(nm, None)
        its = [("train__%s__s%d" % (nm, s), nm, str(tdir / (nm + ext)), self.tl(self.P.size(str(tdir / (nm + ext)))), s, opts, "train")
               for nm in new for s in seeds]
        plan = [(v0h, v0b, it) for it in its] + ([(ch, cb, it) for it in its] if ch != v0h else [])
        if self.C.get("v4") and not self.C["v4"].get("orig_on_new"):
            plan = [(ch, cb, it) for it in its]          # v4: only the committed version's baseline runs
            v0h = ch                                     # a new instance is valid when the committed version solves it
        # same priority as the agent's own measurements (the pool shares cores per campaign and is FIFO inside one, so a
        # refresh queues behind an evaluation in flight). A LOWER priority starves on a saturated pool: on 11-10 a
        # refresh waited more than an hour behind two campaigns' evaluations.
        r2 = self.run_items(plan, "rot_%02d" % rot, prio=5)
        good = [nm for nm in new if all(r2[(v0h, "train__%s__s%d" % (nm, s))]["ok"] for s in seeds)]
        c0, ck = self.cache(v0h), self.cache(ch)
        pairs = []
        for nm in good:
            for s in seeds:
                key = "train__%s__s%d" % (nm, s)
                if key in ck and ck[key]["ok"] and key in c0 and c0[key]["ok"]:
                    pairs.append((ck[key]["cost"], c0[key]["cost"]))
        if self.C.get("v4") and not self.C["v4"].get("orig_on_new"):
            pairs = []                 # the original was not run on the new instances: no out-of-sample record here
        rel = (sum(100.0 * (a - b) / b for a, b in pairs) / len(pairs)) if pairs else None
        retire = old[:len(good)]
        with self.lock:
            self.st["train_order"] = [o for o in order if o not in retire] + good
            self.st["rotations"] = rot
            (self.priv / "train" / "active.json").write_text(json.dumps(self.st["train_order"]))
            for nm in new:
                if nm not in good:
                    self.bks.pop(nm, None)
                    (tdir / (nm + ext)).unlink(missing_ok=True)
            self.load_train()
            wt = self.W / "instances" / "train"
            for nm in retire:
                (wt / (nm + ext)).unlink(missing_ok=True)
            for nm in good:
                shutil.copy(tdir / (nm + ext), wt / (nm + ext))
            with open(self.W / "NOTICES.md", "a") as f:
                f.write("- %s: the training set was refreshed: %d instances were replaced by new ones (`instances/train/` holds the "
                        "current set, `targets.json` the current reference values). All measurements and the commit gate use the "
                        "current set.\n" % (time.strftime("%H:%M"), len(good)))
        self.save()
        self.ledger({"type": "rotation", "rotation": rot, "retired": retire, "added": good, "committed": self.st["version"],
                     "oos_n": len(pairs), "oos_rel_diff_pct": rel, "oos_wins": sum(a < b for a, b in pairs),
                     "oos_losses": sum(a > b for a, b in pairs), "source_version": T.st["accepted"]})

    def rotate_loop(self):
        every = float(self.C["rotate"].get("every_s", 1500))
        while not (self.cd / "STOP").exists():
            if time.time() - self.st.get("last_rotate", self.st.get("t0") or time.time()) >= every:
                try:
                    with self.eval_lock:
                        self.rotate()
                except Exception:
                    with self.lock:
                        self.st["last_rotate"] = time.time()
                    with open(self.cd / "logs" / "judge_errors.log", "a") as ef:
                        ef.write("%s rotate\n%s\n" % (time.strftime("%F %T"), traceback.format_exc()))
            time.sleep(10)

    def finish_batch(self, bname, accepted, rejected, meta, gv, note):
        """Measure a new batch with the original and the committed version BEFORE delivering it (test-then-train: this is
        the committed version's out-of-sample result), then give it to the enhancer and put it into the commit gate."""
        (self.priv / "fresh" / bname / "meta.json").write_text(json.dumps(meta, indent=1))
        v0h, v0b, _ = self.build(self.priv / "versions" / "v0" / "src")
        ch, cb, _ = self.committed_build()
        seeds = self.C.get("seeds", [1])
        opts = self.C.get("train_opts", {})
        its = [("fresh__%s__s%d" % (p.stem, s), p.stem, str(p), self.tl(self.P.size(str(p))), s, opts, "fresh")
               for p in accepted for s in seeds]
        plan = [(v0h, v0b, it) for it in its] + ([(ch, cb, it) for it in its] if ch != v0h else [])
        res = self.run_items(plan, "batch_" + bname, prio=4)
        dead = sorted({it[1] for it in its if not res[(v0h, it[0])]["ok"]})
        for nm in dead:
            (self.priv / "fresh" / bname / "inst" / (nm + Path(accepted[0]).suffix)).unlink(missing_ok=True)
            rejected.append((meta["inst"][nm]["family"], meta["inst"][nm]["n"], "the original solver found no feasible solution in the time limit"))
        dst = self.W / "instances" / "fresh" / bname
        dst.mkdir(parents=True, exist_ok=True)
        kept = sorted((self.priv / "fresh" / bname / "inst").glob(self.P.INSTANCE_GLOB))
        for p in kept:
            shutil.copy(p, dst / p.name)
        with self.lock:
            self.st["batches"].append(bname)
            self.st["active"] = self.active_batches()
            with open(self.W / "NOTICES.md", "a") as f:
                f.write("- %s: new fresh instances delivered in instances/fresh/%s/ (%d files). The commit gate now uses batches: %s.\n" % (
                    time.strftime("%H:%M"), bname, len(kept), ", ".join(self.st["active"])))
        self.save()
        # does the new batch expose that the committed version loses to the original on unseen instances?
        fits = [it for it in self.items("full") if it[6] == "fresh"]
        mk, m0 = self.metrics(fits, self.cache(ch)), self.orig_metrics(fits)
        debt = ch != v0h and self.in_debt(mk, m0)
        if debt:
            with self.lock, open(self.W / "NOTICES.md", "a") as f:
                f.write("- %s: WARNING after batch %s: on the fresh instances now in the commit gate the committed version v%d is "
                        "WORSE than the original solver (mean gap %s%% vs %s%%). An accepted change does not carry over to unseen "
                        "instances. The next commit must repair this (no worse than the original on the fresh instances); a repair "
                        "commit may leave the training-benchmark result unchanged.\n" % (
                            time.strftime("%H:%M"), bname, self.st["version"], fmt(mk["fresh_gap"]), fmt(m0["fresh_gap"])))
        # out-of-sample record of this batch alone (it had never been seen by the enhancer when these runs were made)
        bits = [it for it in its if it[1] not in dead]
        ck, c0 = self.cache(ch), self.cache(v0h)
        bk, b0 = self.metrics(bits, ck), self.metrics(bits, c0)
        pairs = [(ck[it[0]]["cost"], c0[it[0]]["cost"]) for it in bits if it[0] in ck and it[0] in c0 and ck[it[0]]["ok"] and c0[it[0]]["ok"]]
        self.ledger({"type": "fresh_check", "batch": bname, "committed": self.st["version"], "committed_fresh_gap": mk["fresh_gap"],
                     "orig_fresh_gap": m0["fresh_gap"], "debt": bool(debt), "batch_committed_gap": bk["fresh_gap"],
                     "batch_orig_gap": b0["fresh_gap"], "batch_n": len(pairs), "batch_wins": sum(a < b for a, b in pairs),
                     "batch_losses": sum(a > b for a, b in pairs)})
        self.ledger({"type": "batch", "batch": bname, "gen_version": gv, "kept": len(kept), "rejected": len(rejected), "note": note})
        fb = self.feedback()
        return 0, "batch %s accepted: %d instances delivered, %d rejected.\n%s\n\n%s" % (
            bname, len(kept), len(rejected), "\n".join("- rejected %s n=%s: %s" % r for r in rejected[:12]), fb)

    def generate(self, fams, B, rng, bname, outdir, prefix):
        ns = [t[2] for t in self.train]
        lo, hi = min(ns), max(ns)
        outdir.mkdir(parents=True, exist_ok=True)
        tmp = self.G / "out" / bname
        tmp.mkdir(parents=True, exist_ok=True)
        order = list(fams)
        rng.shuffle(order)
        accepted, rejected, meta = [], [], {"inst": {}}
        ext = self.P.INSTANCE_GLOB.replace("*", "")
        for i in range(B):
            fam = order[i % len(order)]
            q = (i + rng.random()) / B
            n = max(lo, min(hi, int(round(ns[min(len(ns) - 1, int(q * len(ns)))] * rng.uniform(0.9, 1.1)))))
            seed = rng.randrange(1, 2**31 - 1)
            name = "%s-%02d-n%d" % (prefix, i + 1, n)
            cmd = jail.jail_cmd(self.cd / "gen", ["python3", "gen/gen.py", "--family", fam, "--n", str(n), "--seed", str(seed),
                                                  "--out", "out/%s/%s%s" % (bname, name, ext)])
            try:
                r = subprocess.run(cmd, env=jail.proot_env(), capture_output=True, text=True, timeout=180)
                err = (r.stderr or r.stdout)[-300:] if r.returncode else ""
            except subprocess.TimeoutExpired:
                err = "generator timed out"
            src = tmp / (name + ext)
            if err or not src.exists():
                rejected.append((fam, n, "generator failed: " + (err or "no output file").strip().replace("\n", " ")[:200]))
                continue
            ok, why = self.P.validate_instance(str(src), lo, hi)
            if ok:
                ok, why = self.in_range(str(src))
            if not ok:
                rejected.append((fam, n, why))
                continue
            shutil.copy(src, outdir / (name + ext))
            accepted.append(outdir / (name + ext))
            meta["inst"][name] = {"family": fam, "n": self.P.size(str(src)), "seed": seed}
        shutil.rmtree(tmp, ignore_errors=True)
        return accepted, rejected, meta

    def in_range(self, path):
        """Fresh instances must be ordinary members of the public benchmark's distribution: every basic feature the
        plugin lists must lie inside the range seen on the public benchmark (5% tolerance on the range)."""
        upper = getattr(self.P, "RANGE_UPPER", [])          # features with an upper bound only
        keys = list(getattr(self.P, "RANGE_FEATURES", [])) + list(upper)
        if not keys:
            return True, ""
        if not hasattr(self, "_frange"):
            fs = [self.P.features(p) for _, p, _ in self.train]
            self._frange = {k: (min(f[k] for f in fs), max(f[k] for f in fs)) for k in keys}
        try:
            f = self.P.features(path)
        except Exception as ex:
            return False, "features: %r" % (ex,)
        for k in keys:
            lo, hi = self._frange[k]
            tol = 0.05 * (hi - lo)
            if (f[k] < lo - tol and k not in upper) or f[k] > hi + tol:
                return False, "%s = %s is outside the public benchmark's range [%s, %s]" % (k, f[k], lo, hi)
        return True, ""

    def active_batches(self):
        live = [b for b in self.st["batches"] if b not in self.st.get("retired", [])]
        return live[-int(self.C.get("active_batches", 2)):]

    def retire(self, batches, reason):
        """Withdraw delivered batches from the gate, the feedback and the enhancer's workspace (operator command; only
        while no judge is serving)."""
        with self.lock:
            r = self.st.setdefault("retired", [])
            for b in batches:
                if b in self.st["batches"] and b not in r:
                    r.append(b)
                    shutil.rmtree(self.W / "instances" / "fresh" / b, ignore_errors=True)
            self.st["active"] = self.active_batches()
            with open(self.W / "NOTICES.md", "a") as f:
                f.write("- %s: fresh batch(es) %s were WITHDRAWN by the operators (%s). They are no longer part of any measurement "
                        "or of the commit gate; ignore earlier results on them. The commit gate now uses batches: %s.\n" % (
                            time.strftime("%H:%M"), ", ".join(batches), reason, ", ".join(self.st["active"]) or "none until the next delivery"))
        self.save()
        self.ledger({"type": "retire", "batches": list(batches), "reason": reason})

    def feedback(self):
        """Per-family aggregates for the instance agent: how the original and the committed version do on its instances."""
        v0h = tree_hash(self.priv / "versions" / "v0" / "src")
        ch = tree_hash(self.priv / "versions" / ("v%d" % self.st["version"]) / "src")
        c0, ck = self.cache(v0h), self.cache(ch)
        fam = {}
        for b in self.st["batches"]:
            if b in self.st.get("retired", []):
                continue
            mp = self.priv / "fresh" / b / "meta.json"
            if not mp.exists():
                continue
            for nm, d in json.loads(mp.read_text())["inst"].items():
                ref = self.fref.get(nm)
                for s in self.C.get("seeds", [1]):
                    k = "fresh__%s__s%d" % (nm, s)
                    a, c = c0.get(k), ck.get(k)
                    if ref and a and a["ok"]:
                        e = fam.setdefault(d["family"], [[], []])
                        e[0].append(100.0 * (a["cost"] - ref) / ref)
                        if c and c["ok"]:
                            e[1].append(100.0 * (c["cost"] - ref) / ref)
        its = self.items("train")
        m0, mk = self.metrics(its, c0), self.metrics(its, ck)
        L = ["## Feedback (committed solver version: v%d)" % self.st["version"],
             "Public benchmark: original version mean gap %s%%, committed version %s%% (improvement %s pp)." % (
                 fmt(m0["train_gap"]), fmt(mk["train_gap"]),
                 fmt(m0["train_gap"] - mk["train_gap"]) if (m0["train_gap"] is not None and mk["train_gap"] is not None) else "n/a"),
             "", "| family | instances | original gap % | committed gap % | improvement pp |", "|---|---|---|---|---|"]
        for f in sorted(fam):
            a, c = fam[f]
            ma = sum(a) / len(a)
            mc = (sum(c) / len(c)) if c else None
            L.append("| %s | %d | %.4f | %s | %s |" % (f, len(a), ma, fmt(mc), fmt(ma - mc) if mc is not None else "n/a"))
        L.append("")
        L.append("(gaps on your instances are relative to the best cost any version has reached on them so far)")
        txt = "\n".join(L)
        (self.G / "feedback").mkdir(parents=True, exist_ok=True)
        (self.G / "feedback" / "latest.md").write_text(txt + "\n")
        return txt

    # ── request loop ─────────────────────────────────────────────────────────
    def handle(self, who, req):
        cmd, a = req.get("cmd"), req.get("args", [])
        if who == "enh":
            if cmd == "leave" and self.C.get("peers") and self.C.get("role") in ("main", "generalist"):
                import islands
                name, reason = (a[0] if a else ""), " ".join(a[1:]).strip()
                items = {it["name"]: it for it in islands._offer_items(self.cd.name)[0]}
                it = items.get(name)
                if it is None:
                    return 2, "usage: ctl.py leave <offer> \"<reason>\"   (offers: %s)" % ", ".join(sorted(items))
                if it["cm"]:
                    return 2, "%s is already part of your committed version." % name
                if it["left"]:
                    return 2, "%s is already closed." % name
                if not any(r.get("suite") == "full" for r in it["ev"]):
                    return 2, ("%s cannot be closed: the ledger has no full evaluation of yours of this offer (an evaluation "
                               "counts for an offer when its note names it, e.g. \"offer %s: ...\"). A quick test on a few "
                               "instances does not show what an offer does over the whole training set." % (name, name))
                if len(reason) < 60:
                    return 2, ("Give the reason: what your measurements show about why it cannot pay off on your version, "
                               "with the variants you measured.")
                self.ledger({"type": "offer_left", "offer": name, "reason": reason, "evals": len(it["ev"]),
                             "best": min(r["paired"]["mean"] for r in it["ev"])})
                islands.write_offer_index(self.cd.name)
                q = islands.offer_queue(self.cd.name)
                return 0, "Recorded: %s is closed (offers/INDEX.md is updated). Next open offer: %s." % (name, q[0] if q else "none")
            if cmd == "claim" and self.C.get("peers"):
                text = " ".join(a).strip()
                if len(text) < 15:
                    return 2, "usage: ctl.py claim \"<one sentence: the direction you are about to work on>\""
                import islands
                m = re.match(r"\s*offer\s+([A-Za-z0-9]+_v\d+)", text)
                if self.C.get("role") in ("main", "generalist"):
                    q = islands.offer_queue(self.cd.name)       # offers with an established effect: first, in order
                    if q and not (m and m.group(1) == q[0]):
                        return 2, ("Not declared. An offer whose effect is established in the scout's project is open: %s "
                                   "(see `offers/INDEX.md`). It comes before anything else: declare it (`claim \"offer %s: "
                                   "...\"`) and work on it. It is closed by a commit of yours that contains it or, when "
                                   "your measurements show why it cannot pay off on your version, by `python3 tools/ctl.py "
                                   "leave %s \"<that reason, with the variants you measured>\"`; after that other "
                                   "directions are open to you again." % (q[0], q[0], q[0]))
                out = islands.claim(self.cd.name, text)
                with self.lock:
                    self.st["claimed"] = True
                    self.st["claim_rule"] = islands.CLAIM_RULE
                self.save()
                return 0, out
            if cmd in ("probe", "eval") and self.C.get("peers"):
                import islands
                if self.st.get("claimed") and self.st.get("claim_rule") != islands.CLAIM_RULE:
                    with self.lock:                 # declared under an older version of the board: declare again
                        self.st["claimed"] = False
                    self.save()
                if not self.st.get("claimed"):
                    return 2, ("Declare the direction first: `python3 tools/ctl.py claim \"<one sentence: what you are about "
                               "to work on>\"`. The answer lists what the other projects are working on now and everything "
                               "they have already declared, tried, kept or left; if yours is the same idea or a variant of "
                               "any of it, choose another. (A declaration is needed before the first evaluation, again after "
                               "every commit, and again when the board's rules have changed - which is the case if you had "
                               "declared and see this message.)")
            if cmd == "probe" and self.C.get("v4"):
                known = {t[0] for t in self.train}
                names = [x for x in a if x in known or (" " not in x and "-n" in x)]
                with self.eval_lock:
                    return self.do_probe4(names, " ".join(x for x in a if x not in names))
            if cmd == "eval" and self.C.get("v4"):
                with self.eval_lock:
                    return self.do_eval4(a[0] if a else "screen", " ".join(a[1:]))
            if cmd == "commit" and self.C.get("v4"):
                with self.eval_lock:
                    rc, txt = self.do_commit4(" ".join(a))
                return self.after_decision(rc, txt)
            if cmd == "eval":
                return self.do_eval(a[0] if a else "dev", " ".join(a[1:]))
            if cmd == "commit":
                return self.do_commit(" ".join(a))
            if cmd == "revert":
                rc, txt = self.do_revert()
                return self.after_decision(rc, txt)
            if cmd == "status":
                return self.do_status()
            if cmd == "run":
                seed = 1
                names = []
                i = 0
                long = "--long" in a
                a = [x for x in a if x != "--long"]
                while i < len(a):
                    if a[i] == "--seed" and i + 1 < len(a):
                        seed = int(a[i + 1])
                        i += 2
                        continue
                    names.append(a[i])
                    i += 1
                if len(names) == 2 and names[1].isdigit():     # old form: run <instance> <seed>
                    seed, names = int(names[1]), names[:1]
                return self.do_run(names, seed, long)
            return 2, "commands: status | run <instance> [<instance> ...] [--seed k] [--long] | probe <instance> ... \"note\" | eval %s|full \"note\" | commit \"note\" | revert" % (
                "screen" if self.C.get("v4") else "dev")
        if cmd == "submit":
            return self.do_submit(" ".join(a))
        if cmd == "feedback":
            return 0, self.feedback()
        return 2, "commands: submit \"note\" | feedback"

    def loop(self, who, box):
        (box / "req").mkdir(parents=True, exist_ok=True)
        (box / "res").mkdir(parents=True, exist_ok=True)
        while not (self.cd / "STOP").exists():
            for f in sorted((box / "req").glob("*.json")):
                try:
                    req = json.loads(f.read_text())
                except Exception:
                    continue
                t0 = time.time()
                try:
                    rc, txt = self.handle(who, req)
                    if who == "enh":
                        try:                       # tell the agent once about notices it has not been told about
                            sz = (self.W / "NOTICES.md").stat().st_size
                        except OSError:
                            sz = 0
                        if sz > self.st.get("notices_told", 0):
                            self.st["notices_told"] = sz
                            self.save()
                            txt = (txt or "").rstrip("\n") + "\n\n[NOTICES.md has new entries since your last request - read it.]\n"
                except Exception:
                    rc, txt = 9, "judge error (reported to the operators); try again later\n"
                    with open(self.cd / "logs" / "judge_errors.log", "a") as ef:
                        ef.write("%s %s %s\n%s\n" % (time.strftime("%F %T"), who, req, traceback.format_exc()))
                tmp = box / "res" / (f.stem + ".tmp")
                tmp.write_text(json.dumps({"rc": rc, "text": txt}))
                os.replace(tmp, box / "res" / (f.stem + ".json"))
                f.unlink(missing_ok=True)
                with open(self.cd / "logs" / "judge.log", "a") as lf:
                    lf.write("%s %s %s rc=%s %.0fs\n" % (time.strftime("%F %T"), who, json.dumps(req)[:200], rc, time.time() - t0))
            time.sleep(0.5)

    def after_decision(self, rc, txt):
        """After a commit or a revert: the take-over of the main island's version that was put off at the last session
        boundary because an evaluation of the previous session had passed the gate and was not committed (apply_sync)
        happens now, before the answer goes back, so that the agent never works on a tree that is about to be replaced."""
        if rc != 0 or not self.st.get("sync_deferred"):
            return rc, txt
        with self.lock:
            self.st["sync_deferred"] = None
        self.save()
        try:
            (self.cd / "SYNC").write_text(json.dumps({"id": "deferred_%d" % int(time.time())}))
            res = self.apply_sync() or {}
        except Exception:
            res = {"error": traceback.format_exc()[-800:]}
        if res.get("done"):
            txt = (txt or "").rstrip("\n") + (
                "\n\nAS ANNOUNCED IN NOTICES.md, THE SERVICE HAS NOW SET YOUR COMMITTED VERSION TO THE MAIN PROJECT'S CURRENT VERSION "
                "(its v%s; here v%s): `src/` and `committed/` hold it. Declare your next direction before the next evaluation.\n"
                % (res.get("main_version"), res.get("version")))
        else:
            txt = (txt or "").rstrip("\n") + ("\n\n(The announced change to the main project's version was not made now: %s. "
                                              "It is tried again at the start of your next session.)\n"
                                              % (res.get("skipped") or "service error").replace("\n", " ")[:200])
        return rc, txt

    def apply_rollback(self):
        """`camp/<name>/ROLLBACK` = {"to": N, "reason": "..."}: the committed version becomes a new version with the tree
        of v N; the versions after N are withdrawn (kept on disk and in the ledger)."""
        f = self.cd / "ROLLBACK"
        if not f.exists():
            return None
        try:
            req = json.loads(f.read_text())
            to, cur = int(req["to"]), int(self.st["version"])
            src = self.priv / "versions" / ("v%d" % to) / "src"
            if not (0 <= to < cur) or not src.exists():
                raise ValueError("cannot go back to v%d from v%d" % (to, cur))
            h, v = tree_hash(src), cur + 1
            dropped = list(range(to + 1, cur + 1))
            notes = [r.get("note") or "" for r in self.ledger_rows() if r.get("type") == "commit" and r.get("version") in dropped]
            offers = sorted(set(re.findall(r"\b[A-Z]\d+_v\d+\b", " ".join(notes))))
            with self.lock:
                copy_tree(src, self.priv / "versions" / ("v%d" % v) / "src")
                copy_tree(src, self.W / "committed")
                self.st["version"], self.st["hash"] = v, h
                self.st["last_commit_attempt"] = "the service set the committed version back to the tree of v%d (now v%d) at %s" % (
                    to, v, time.strftime("%H:%M"))
                self.st["claimed"] = False
                self.st["last_eval"] = None             # an evaluation against the withdrawn version no longer counts
                if self.paired:                         # v5: the re-instated tree gets a fresh seed block
                    self.st["block"] = int(self.st.get("block", 0)) + 1
            self.save()
            self.ledger({"type": "rollback", "version": v, "to": to, "dropped": dropped, "hash": h, "reason": req.get("reason"),
                         "offers_reopened": offers})
            with open(self.W / "NOTICES.md", "a") as nf:
                nf.write("- %s: THE COMMITTED VERSION WAS SET BACK BY THE SERVICE: v%d is the tree of v%d; %s withdrawn. Reason: %s "
                         "`committed/` holds v%d. Your `src/` still contains the withdrawn changes: run `python3 tools/ctl.py "
                         "revert` (it resets `src/` to the committed version) and re-apply the change you are working on to "
                         "that tree; evaluations are compared with v%d from now on.%s\n" % (
                             time.strftime("%H:%M"), v, to, ", ".join("v%d" % d for d in dropped) + (" is" if len(dropped) == 1 else " are"),
                             (req.get("reason") or "").rstrip() or "-", v, v,
                             (" The offers that were part of them (%s) are open again in `offers/INDEX.md`." % ", ".join(offers)) if offers else ""))
            f.rename(self.cd / ("ROLLBACK.done_v%d" % v))
            return v
        except Exception:
            with open(self.cd / "logs" / "judge_errors.log", "a") as ef:
                ef.write("%s rollback: %s\n" % (time.strftime("%F %T"), traceback.format_exc()))
            try:
                f.rename(self.cd / "ROLLBACK.failed")
            except OSError:
                pass
            return None

    def apply_sync(self):
        """`camp/<name>/SYNC` = {"id": ...}, written by the driver of a secondary island between two sessions (camp
        option sync_main): this island's committed version becomes the MAIN island's committed version, as a new
        version of its own line (ledger record "rebase"; never an offer). Why (job-shop run of 11-13): the secondaries
        kept working on the original plus their own few changes while the main island's version ran twice as many
        iterations, so what they measured did not hold on the main island's version and every offer had to be ported.
        Not done (answer "skipped") when the main island has nothing new, and - so that a secondary keeps building on
        its own committed changes until the main island has decided about them - while one of its offers is still
        open, at most sync_max_skips session boundaries in a row. The main island's runs of that version (same
        training instances, same machine) are copied, so the new baseline needs no runs when the seed block is the
        main island's; the work tree the last session left is kept in before_sync/. The answer is written to
        priv/sync_result.json."""
        f = self.cd / "SYNC"
        if not f.exists():
            return None
        res = {"t": time.time(), "done": False}
        got_lock = False
        try:
            import islands
            try:
                res["id"] = json.loads(f.read_text()).get("id")
            except ValueError:
                res["id"] = None
            main = next((p for p in self.C.get("peers", [])
                         if (islands._json(islands.SE / "camp" / p / "camp.json", {}) or {}).get("role") in ("main", "generalist")), None)
            if not main:
                raise ValueError("no main island among the peers")
            M = islands.SE / "camp" / main
            lab = islands._label(self.cd.name)
            WRITING = "the main project's version is being written"

            def read_main():
                ms_ = islands._json(M / "priv" / "state.json") or {}
                mv_, mh_ = int(ms_.get("version", 0)), ms_.get("hash")
                src_ = M / "priv" / "versions" / ("v%d" % mv_) / "src"
                items, _ = islands._offer_items(main)
                known = {it["name"] for it in items}
                own = ["%s_v%d" % (lab, r["version"]) for r in self.ledger_rows() if r.get("type") == "commit"]
                op = [it["name"] for it in items if it["name"].startswith(lab + "_v") and not it["cm"] and not it["left"]]
                op += [n for n in own if n not in known]        # committed here, not yet registered over there
                cl = [it["name"] for it in items if it["name"].startswith(lab + "_v") and it["left"] and not it["cm"]]
                return ms_, mv_, mh_, src_, op, cl

            def no_take_over(mv_, mh_, src_):
                if mv_ < 1 or not src_.exists():
                    return "the main project has no committed version yet"
                if mh_ == self.st["hash"]:
                    return "already on the main project's version"
                if self.st.get("sync_main_hash") == mh_:
                    return "the main project has not committed since the last synchronisation"
                if tree_hash(src_) != mh_:
                    return WRITING
                return None

            ms, mv, mh, src, open_, closed = read_main()
            why = no_take_over(mv, mh, src)
            if why is None and open_ and int(self.st.get("sync_skips", 0)) < int(self.C.get("sync_max_skips", 1)):
                why = "offers still open at the main project: %s" % ", ".join(open_)
                with self.lock:
                    self.st["sync_skips"] = int(self.st.get("sync_skips", 0)) + 1
                self.save()
            if why is None or why == WRITING:
                got_lock = self.eval_lock.acquire(timeout=float(self.C.get("sync_wait_s", 600)))
                if not got_lock:
                    why = "an evaluation of the previous session is still running"
                else:
                    # (11-16) read the main island again now: this island may have waited here for an evaluation of its
                    # own for most of an hour, and the main island may have committed meanwhile (the independent-set
                    # run: a secondary took v9 in the very seconds v10 was committed)
                    for _ in range(8):
                        ms, mv, mh, src, open_, closed = read_main()
                        why = no_take_over(mv, mh, src)
                        if why != WRITING:
                            break
                        time.sleep(1)
            if why is None:
                # (11-16) the last evaluation of the session that just ended passed the commit gate and its tree is still
                # the work tree: a take-over now would replace that tree and the result would never be committed, i.e.
                # never offered to the main island (it happened twice in the independent-set run). The next session is
                # told to commit or revert first; the take-over follows that decision at once (after_decision). Put
                # off once only: an agent that ignores the notice is taken over at the next boundary.
                le = self.st.get("last_eval") or {}
                wt = tree_hash(self.W / "src") if (self.W / "src").exists() else None
                prev = self.st.get("sync_deferred") or {}
                if (le.get("suite") == "full" and le.get("complete") and le.get("gate") and le.get("hash") == wt
                        and wt != self.st["hash"] and le.get("committed_hash") == self.st["hash"] and prev.get("eval") != le.get("id")):
                    why = ("evaluation eval_%03d of the previous session passed the commit gate and is not committed "
                           "(commit or revert comes first; the take-over follows it at once)" % le["id"])
                    with self.lock:
                        self.st["sync_deferred"] = {"t": time.time(), "eval": le["id"], "main_version": mv}
                    self.save()
                    with open(self.W / "NOTICES.md", "a") as nf:
                        nf.write("- %s: THE LAST EVALUATION OF YOUR PREVIOUS SESSION (eval_%03d) PASSED THE COMMIT GATE and its tree is "
                                 "still your `src/` (its summary: `runs/eval_%03d/summary.md`). Decide it before anything else: "
                                 "`python3 tools/ctl.py commit \"<note>\"` keeps it (it then goes to the main project as an "
                                 "offer), `python3 tools/ctl.py revert` drops it. Right after either, the service sets your "
                                 "committed version to the main project's current version (its v%d), which it would otherwise "
                                 "have done at the start of this session.\n" % (time.strftime("%H:%M"), le["id"], le["id"], mv))
            if why:
                res["skipped"] = why
                self.ledger({"type": "sync_skipped", "why": why, "main_version": mv})
            else:
                with self.lock:
                    old_v, old_h = int(self.st["version"]), self.st["hash"]
                    v = old_v + 1
                    copy_tree(src, self.priv / "versions" / ("v%d" % v) / "src")
                    if (self.W / "src").exists():
                        (self.W / "before_sync").mkdir(exist_ok=True)
                        copy_tree(self.W / "src", self.W / "before_sync" / ("v%d_worktree" % old_v))
                    copy_tree(src, self.W / "src")
                    copy_tree(src, self.W / "committed")
                    mc = islands._json(M / "priv" / "cache" / (mh + ".json"), {}) or {}
                    known_inst = {t[0] for t in self.train}
                    recs = [r for k, r in mc.items() if isinstance(r, dict) and r.get("key") == k and k.split("__")[1] in known_inst]
                    if recs:
                        self.cache_put(mh, recs)
                    ob, mb = int(self.st.get("block", 0)), int(ms.get("block", 0))
                    self.st.update(version=v, hash=mh, last_eval=None, claimed=False, sync_main_hash=mh, sync_main_version=mv, sync_skips=0,
                                   sync_deferred=None,
                                   last_commit_attempt="the service set the committed version to the main project's v%d (here v%d) at %s" % (
                                       mv, v, time.strftime("%H:%M")))
                    if self.paired:                 # the main island's seed block when that is a new one here (its runs are in the cache)
                        self.st["block"] = mb if mb > ob else ob + 1
                        self.st["block_spent"] = False
                self.save()
                self.ledger({"type": "rebase", "version": v, "from": main, "from_version": mv, "hash": mh, "prev_version": old_v,
                             "prev_hash": old_h, "block": self.st.get("block", 0), "main_block": mb, "runs_copied": len(recs),
                             "open_offers": open_, "closed_offers": closed})
                msg = ("- %s: YOUR COMMITTED VERSION WAS SET TO THE MAIN PROJECT'S CURRENT VERSION BY THE SERVICE (its v%d; here it "
                       "is v%d): `committed/` and `src/` hold it now. It contains everything the main project has kept, "
                       "including what it took from your offers. %sYour work tree as the last session left it is in "
                       "`before_sync/v%d_worktree/` (for reference: a change you were working on has to be re-applied to "
                       "the new `src/` by hand). Evaluations are compared with v%d from now on; its baseline runs are "
                       "provided by the service (same instances, same machine). Everything measured before this point - "
                       "yours and what `NOTES.md` records - was measured on your previous version (v%d).\n" % (
                           time.strftime("%H:%M"), mv, v,
                           ("Not in it: your changes that the main project has not taken (%s%s). " % (
                               ("still open there, it is working on them - do not re-apply them: " + ", ".join(open_)) if open_ else "",
                               ("; " if open_ and closed else "") + (("closed there: " + ", ".join(closed)) if closed else ""))) if (open_ or closed) else "",
                           old_v, v, old_v))
                with open(self.W / "NOTICES.md", "a") as nf:
                    nf.write(msg)
                try:
                    with open(self.W / "NOTES.md", "a") as nf:
                        nf.write("\n\n---- %s (service): the committed version was set to the main project's v%d (here v%d). Everything "
                                 "above this line was measured on the previous version (v%d). ----\n\n" % (time.strftime("%H:%M"), mv, v, old_v))
                except OSError:
                    pass
                res.update(done=True, version=v, main_version=mv, block=self.st.get("block", 0), runs_copied=len(recs))
        except Exception:
            res["error"] = traceback.format_exc()[-1500:]
            with open(self.cd / "logs" / "judge_errors.log", "a") as ef:
                ef.write("%s sync: %s\n" % (time.strftime("%F %T"), traceback.format_exc()))
        finally:
            if got_lock:
                self.eval_lock.release()
            try:
                tmp = self.priv / "sync_result.json.tmp"
                tmp.write_text(json.dumps(res))
                os.replace(tmp, self.priv / "sync_result.json")
            except OSError:
                pass
            f.unlink(missing_ok=True)
        if res.get("done"):
            self.baseline_async(res["version"])     # only the runs that are missing; writes the "baseline" record
        return res

    def ledger_rows(self):
        out = []
        try:
            for line in open(self.priv / "ledger.jsonl"):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        except OSError:
            pass
        return out

    def serve(self):
        (self.cd / "logs").mkdir(exist_ok=True)
        import fcntl
        self._lockf = open(self.priv / "judge.lock", "w")
        try:
            fcntl.flock(self._lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)     # exactly one serving judge per campaign
        except OSError:
            print("another judge is serving this campaign; exiting")
            return
        if not self.st.get("t0"):
            self.st["t0"] = time.time()
            self.save()
        self.apply_rollback()
        # RESUME: the campaign clock (t0, the rotation timer) counts only the time a judge was serving. `alive` is
        # touched every 20 s; a gap means the campaign was down, and the clocks are shifted by it.
        af = self.priv / "alive"
        try:
            gap = time.time() - float(af.read_text())
        except (OSError, ValueError):
            gap = 0.0
        if gap > 90:
            self.st["t0"] += gap
            if self.st.get("last_rotate"):
                self.st["last_rotate"] += gap
            if self.st.get("last_draw"):
                self.st["last_draw"] += gap
            self.save()
            self.ledger({"type": "resume", "downtime_s": gap})
        af.write_text(str(time.time()))
        th = [threading.Thread(target=self.loop, args=("enh", self.W / "box"), daemon=True)]
        if self.C.get("rotate") and self.C.get("train_twin"):
            th.append(threading.Thread(target=self.rotate_loop, daemon=True))
        if self.C.get("fresh") and self.C.get("twin"):
            th.append(threading.Thread(target=self.batch_loop, daemon=True))
        elif self.C.get("fresh"):
            th.append(threading.Thread(target=self.loop, args=("gen", self.G / "box"), daemon=True))
        for t in th:
            t.start()
        beat = 0.0
        while not (self.cd / "STOP").exists():
            time.sleep(2)
            if (self.cd / "SYNC").exists():         # a secondary island between two sessions (see apply_sync)
                self.apply_sync()
            if time.time() - beat > 20:
                beat = time.time()
                try:
                    af.write_text(str(beat))
                except OSError:
                    pass

    # ── setup-time commands ──────────────────────────────────────────────────
    def calib(self, seeds=(1, 2, 3)):
        if self.C.get("v4") and len(self.C["v4"].get("base_seeds", [])) > 3:
            seeds = tuple(self.C["v4"]["base_seeds"])      # the original's baseline = the calibration runs
        v0 = self.priv / "versions" / "v0" / "src"
        h, bindir, err = self.build(v0)
        if err:
            print("BUILD FAILED\n" + err)
            return 1
        opts = self.C.get("train_opts", {})
        its = [("train__%s__s%d" % (name, s), name, path, self.tl(n), s, opts, "train")
               for name, path, n in self.train for s in seeds]
        c = self.cache(h)
        plan = [(h, bindir, it) for it in its if it[0] not in c]
        t0 = time.time()
        self.run_items(plan, "calib")
        c = self.cache(h)
        means = []
        for s in seeds:
            sub = [it for it in its if it[4] == s]
            m = self.metrics(sub, c)
            means.append(m["train_gap"])
            print("seed %d: mean gap %s%% hits %d/%d fails %d" % (s, fmt(m["train_gap"]), m["hits"], m["n_train"], m["fails"]))
        if any(m is None for m in means):
            bad = [r for r in c.values() if not r["ok"]][:3]
            print("CALIBRATION FAILED: the original solver produced no valid result; first failures: %s" % [
                (r["name"], r["msg"], r["rc"]) for r in bad])
            if bad and os.path.exists(bad[0]["log"]):
                print(open(bad[0]["log"], errors="ignore").read()[-600:])
            return 1
        mu = sum(means) / len(means)
        sd = (sum((x - mu) ** 2 for x in means) / max(1, len(means) - 1)) ** 0.5
        over = max((c[it[0]]["elapsed"] - it[3]) for it in its)
        print("mean over seeds %.4f%%  sd of the seed means %.4f pp  max overrun %.1fs  wall %.0fs" % (mu, sd, over, time.time() - t0))
        self.st.update({"hash": h, "eps_train": round(sd, 4), "eps_fresh": round(2 * sd, 4), "v0_seed_means": means})
        self.save()
        self.ledger({"type": "calib", "hash": h, "means": means, "sd": sd})
        return 0

    def seal(self, k):
        gv = self.st["gen_version"]
        gsrc = self.priv / "gen" / ("v%d" % gv)
        if not gsrc.exists():
            print("no accepted generator")
            return 1
        copy_tree(gsrc, self.G / "gen")
        fams = [f["name"] for f in json.loads((gsrc / "families.json").read_text())]
        rng = random.Random("%s-sealed-%d" % (self.C.get("secret", "s"), time.time_ns()))
        acc, rej, meta = self.generate(fams, k, rng, "sealed", self.priv / "sealed" / "inst", "S")
        (self.priv / "sealed" / "meta.json").write_text(json.dumps(meta, indent=1))
        print("sealed set: %d accepted, %d rejected" % (len(acc), len(rej)))
        return 0


if __name__ == "__main__":
    J = Judge(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "serve"
    if mode == "serve":
        J.serve()
    elif mode == "calib":
        sys.exit(J.calib())
    elif mode == "seal":
        sys.exit(J.seal(int(sys.argv[3])))
    elif mode == "retire":            # judge.py <camp> retire "<reason>" b01 [b02 ...]
        import fcntl
        lf = open(J.priv / "judge.lock", "w")
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)   # refuses to run while a judge is serving
        J.retire(sys.argv[4:], sys.argv[3])
        print("retired:", J.st.get("retired"), "active:", J.st["active"])
