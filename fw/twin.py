#!/usr/bin/env python3
"""Benchmark twin (phase 0): an imitator agent writes a generator of ordinary instances of the public benchmark's
kind; a discriminator agent tries to tell generated files from public ones; this service generates the samples,
builds the blind exam, scores the discriminator, checks solver behaviour, and decides when the twin is certified.

  python3 twin.py <twin_dir> serve            request loop for the two agents
  python3 twin.py <twin_dir> reissue          new exam from the current generator version (no generator change)
  python3 twin.py <twin_dir> status

Certified = feature ranges ok, no copies of public files, the discriminator's exam accuracy <= acc_threshold in two
consecutive rounds, and the original solver behaves the same on generated and public instances (two-sample test).
"""
import concurrent.futures
import importlib
import json
import math
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
from judge import copy_tree   # noqa: E402

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    a, b = sorted(a), sorted(b)
    if not a or not b:
        return 0.0, 1.0
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x:
            i += 1
        while j < len(b) and b[j] <= x:
            j += 1
        d = max(d, abs(i / float(len(a)) - j / float(len(b))))
    en = math.sqrt(len(a) * len(b) / float(len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.3:                      # the series does not converge near 0; the true value is 1 to 5 decimals
        return d, 1.0
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, max(0.0, min(1.0, p))


def auc(pos, neg):
    if not pos or not neg:
        return 0.5
    w = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return w / (len(pos) * len(neg))


class Twin:
    def __init__(self, tdir):
        self.td = Path(tdir).resolve()
        self.C = json.loads((self.td / "twin.json").read_text())
        self.P = importlib.import_module("problems." + self.C["problem"])
        self.priv = self.td / "priv"
        self.G = self.td / "gen" / "work"
        self.D = self.td / "dis" / "work"
        for sub in ("gen", "exams", "tells", "reasons", "behaviour", "bin"):
            (self.priv / sub).mkdir(parents=True, exist_ok=True)
        (self.td / "logs").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        sp = self.priv / "state.json"
        self.st = json.loads(sp.read_text()) if sp.exists() else {"gen_version": 0, "accepted": 0, "round": 0, "rounds": [],
                                                                  "certified": False}
        self.ext = self.P.INSTANCE_GLOB.replace("*", "")
        self.train = sorted(((p.stem, str(p), self.P.size(str(p))) for p in Path(self.C["train_dir"]).glob(self.P.INSTANCE_GLOB)),
                            key=lambda t: (t[2], t[0]))
        self.thr = float(self.C.get("acc_threshold", 0.60))
        self.max_submits = int(self.C.get("max_submits", 3))
        splitp = self.priv / "split.json"
        if splitp.exists():
            self.split = json.loads(splitp.read_text())
        else:                               # stratified by size: of each consecutive pair, one to study, one to exam
            rng = random.Random(self.C.get("secret", "s") + "-split")
            study, exam = [], []
            for i in range(0, len(self.train), 2):
                pair = [t[0] for t in self.train[i:i + 2]]
                rng.shuffle(pair)
                study.append(pair[0])
                if len(pair) > 1:
                    exam.append(pair[1])
            self.split = {"study": study, "exam": exam}
            splitp.write_text(json.dumps(self.split))
        self._sigs = None
        self._frange = None

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self):
        with self.lock:
            tmp = self.priv / "state.json.tmp"
            tmp.write_text(json.dumps(self.st, indent=1))
            os.replace(tmp, self.priv / "state.json")

    def ledger(self, rec):
        rec["t"] = time.time()
        with self.lock, open(self.priv / "ledger.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ── checks ───────────────────────────────────────────────────────────────
    def static_check(self, d, what):
        """Generators and discriminator programs must be programs, not containers of data."""
        tot, nums = 0, 0
        for p in Path(d).rglob("*"):
            if p.is_file():
                tot += p.stat().st_size
                if p.suffix == ".py":
                    t = p.read_text(errors="ignore")
                    nums += len(NUM.findall(t))
                    if what == "tell" and re.search(r"hashlib|md5|sha1|sha256|crc32|zlib", t):
                        return False, "tell.py must not fingerprint files (hash/checksum functions found)"
        if tot > 400_000:
            return False, "the directory is %d bytes; a program of this kind needs far less (no data files)" % tot
        if nums > 4000:
            return False, "the code contains %d numeric literals: stored data tables are not allowed" % nums
        return True, ""

    def frange(self):
        if self._frange is None:
            keys = list(getattr(self.P, "RANGE_FEATURES", [])) + list(getattr(self.P, "RANGE_UPPER", []))
            fs = [self.P.features(p) for _, p, _ in self.train]
            self._frange = {k: (min(f[k] for f in fs), max(f[k] for f in fs)) for k in keys}
        return self._frange

    def validate(self, path):
        ns = [t[2] for t in self.train]
        ok, why = self.P.validate_instance(path, min(ns), max(ns))
        if not ok:
            return False, why
        if self.C.get("naive"):               # generator written without the benchmark: only well-formedness is checked
            return True, ""
        f = self.P.features(path)
        upper = getattr(self.P, "RANGE_UPPER", [])
        for k, (lo, hi) in self.frange().items():
            tol = 0.05 * (hi - lo)
            if (f[k] < lo - tol and k not in upper) or f[k] > hi + tol:
                return False, "%s = %s is outside the public benchmark's range [%s, %s]" % (k, f[k], lo, hi)
        if hasattr(self.P, "signature"):
            if self._sigs is None:
                self._sigs = [self.P.signature(p) for _, p, _ in self.train]
            s = self.P.signature(path)
            m = max(self.P.sig_similarity(s, t) for t in self._sigs)
            if m > float(self.C.get("copy_threshold", 0.2)):
                return False, "the instance reproduces a public file (similarity %.2f): copies or perturbed copies of public files are not allowed" % m
        return True, ""

    # ── generation (clean jail: the generator code only, no public files) ─────
    def gen_run(self, gsrc, plan, outdir):
        """One generator run at a time per twin: the work directory `genrun/` is shared (11-11: two campaigns set up at
        the same moment drew from the same twin and one of them crashed)."""
        import fcntl
        with open(self.td / "genrun.lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            return self._gen_run(gsrc, plan, outdir)

    def _gen_run(self, gsrc, plan, outdir):
        """plan: [(name, family, n, seed)] -> {name: (ok, why)}; valid files end up in outdir."""
        home = self.td / "genrun"
        jail.prepare_home(home)
        work = home / "work"
        copy_tree(gsrc, work / "gen")
        tmp = work / "out"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        outdir.mkdir(parents=True, exist_ok=True)

        def one(job):
            name, fam, n, seed = job
            cmd = jail.jail_cmd(home, ["python3", "gen/gen.py", "--family", fam, "--n", str(n), "--seed", str(seed),
                                       "--out", "out/%s%s" % (name, self.ext)], cpus=jail.alloc_cpus("svc/" + self.td.name, cores=2))
            try:
                r = subprocess.run(cmd, env=jail.proot_env(), capture_output=True, text=True, timeout=240)
                err = (r.stderr or r.stdout)[-300:] if r.returncode else ""
            except subprocess.TimeoutExpired:
                err = "generator timed out"
            src = tmp / (name + self.ext)
            if err or not src.exists():
                return name, (False, "generator failed: " + (err or "no output file").strip().replace("\n", " ")[:200])
            try:
                ok, why = self.validate(str(src))
            except Exception as ex:
                ok, why = False, "unreadable: %r" % (ex,)
            if ok:
                if hasattr(self.P, "canonical"):      # fixed layout, no name / comment lines: the file carries no hint of
                    self.P.canonical(str(src), str(outdir / (name + self.ext)))    # where its recipe comes from
                else:
                    shutil.copy(src, outdir / (name + self.ext))
            return name, (ok, why)

        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            res = dict(ex.map(one, plan))
        shutil.rmtree(tmp, ignore_errors=True)
        return res

    def sample(self, gsrc, fams, rng, outdir, tag):
        """One generated instance per public instance, with the SAME size (size-matched twin sample)."""
        fams = list(fams)
        rng.shuffle(fams)
        todo = [(name, n) for name, _, n in self.train]
        meta, rejected = {}, []
        for attempt in range(3):
            plan = []
            for i, (name, n) in enumerate(todo):
                plan.append(("%s__%s" % (tag, name), fams[(i + attempt) % len(fams)], n, rng.randrange(1, 2**31 - 1)))
            res = self.gen_run(gsrc, plan, outdir)
            nxt = []
            for (name, n), job in zip(todo, plan):
                ok, why = res[job[0]]
                if ok:
                    meta[name] = {"file": job[0] + self.ext, "family": job[1], "n": n, "seed": job[3]}
                else:
                    rejected.append((job[1], n, why))
                    nxt.append((name, n))
            todo = nxt
            if not todo:
                break
        return meta, rejected, [t[0] for t in todo]

    # ── imitator requests ────────────────────────────────────────────────────
    def do_gen_submit(self, note):
        gdir = self.G / "gen"
        if not (gdir / "gen.py").exists() or not (gdir / "families.json").exists():
            return 2, "submit needs gen/gen.py and gen/families.json"
        try:
            fams = [f["name"] for f in json.loads((gdir / "families.json").read_text())]
            assert fams and all(isinstance(x, str) and x.replace("_", "").isalnum() for x in fams)
        except Exception:
            return 2, "gen/families.json must be a list of {\"name\": <alphanumeric id>, \"description\": ...}"
        ok, why = self.static_check(gdir, "gen")
        if not ok:
            return 1, "generator rejected: " + why
        with self.lock:
            self.st["gen_version"] += 1
            gv = self.st["gen_version"]
        snap = self.priv / "gen" / ("v%d" % gv)
        copy_tree(gdir, snap / "code")
        rng = random.Random("%s-gen-%d-%d" % (self.C.get("secret", "s"), gv, time.time_ns()))
        meta, rejected, missing = self.sample(snap / "code", fams, rng, snap / "sample", "g%d" % gv)
        nrej = len(rejected)
        if len(missing) > 0.1 * len(self.train) or nrej > 0.3 * len(self.train):
            self.ledger({"type": "gen_rejected", "gen_version": gv, "missing": len(missing), "rejected": nrej})
            why = {}
            for fam, n, w in rejected:
                why.setdefault((fam, w.split(" = ")[0][:90]), []).append(n)
            return 1, "generator rejected: %d of %d requested instances could not be produced validly (%d invalid attempts).\n%s" % (
                len(missing), len(self.train), nrej,
                "\n".join("- family %s: %s (%d times, e.g. n=%s)" % (k[0], k[1], len(v), v[0]) for k, v in sorted(why.items())[:15]))
        (snap / "meta.json").write_text(json.dumps(meta, indent=1))
        with self.lock:
            self.st["accepted"] = gv
        self.save()
        if self.C.get("naive"):
            self.ledger({"type": "gen_accepted", "gen_version": gv, "invalid_attempts": nrej, "families": fams, "note": note})
            return 0, "generator version %d accepted (%d invalid attempts were re-drawn)." % (gv, nrej)
        txt = self.issue_exam(gv, meta)
        self.ledger({"type": "gen_accepted", "gen_version": gv, "invalid_attempts": nrej, "families": fams, "note": note})
        return 0, "generator version %d accepted (%d invalid attempts were re-drawn).\n%s" % (gv, nrej, txt)

    def issue_exam(self, gv, meta=None):
        """Build the discriminator's round: canonical, disguised, size-matched study and exam files."""
        snap = self.priv / "gen" / ("v%d" % gv)
        if meta is None:
            meta = json.loads((snap / "meta.json").read_text())
        with self.lock:
            self.st["round"] += 1
            k = self.st["round"]
        rng = random.Random("%s-exam-%d-%d" % (self.C.get("secret", "s"), k, time.time_ns()))
        pub = {name: path for name, path, _ in self.train}
        for d in ("study", "exam"):
            shutil.rmtree(self.D / d, ignore_errors=True)
        (self.D / "study" / "P").mkdir(parents=True)
        (self.D / "study" / "Q").mkdir(parents=True)
        (self.D / "exam").mkdir(parents=True)
        idx = list(range(1, 2 * len(self.split["study"]) + 1))
        rng.shuffle(idx)
        it = iter(idx)
        for name in self.split["study"]:
            if name in meta:
                self.P.canonical(pub[name], str(self.D / "study" / "P" / ("s%03d%s" % (next(it), self.ext))), rng)
                self.P.canonical(str(snap / "sample" / meta[name]["file"]), str(self.D / "study" / "Q" / ("s%03d%s" % (next(it), self.ext))), rng)
        labels = {}
        ex = [(name, "P") for name in self.split["exam"] if name in meta] + [(name, "Q") for name in self.split["exam"] if name in meta]
        rng.shuffle(ex)
        for i, (name, lab) in enumerate(ex):
            fn = "e%03d%s" % (i + 1, self.ext)
            src = pub[name] if lab == "P" else str(snap / "sample" / meta[name]["file"])
            self.P.canonical(src, str(self.D / "exam" / fn), rng)
            labels[fn] = {"label": lab, "public_name": name}
        (self.priv / "exams" / ("r%d.json" % k)).write_text(json.dumps(labels))
        old = []
        for tp in sorted((self.priv / "tells").glob("*.py")):
            acc, a, _ = self.score_tell(tp, labels)
            old.append({"program": tp.stem, "acc": acc, "auc": a})
        rd = {"round": k, "gen_version": gv, "best_acc": None, "best_auc": None, "submits": 0, "old_tells": old, "behaviour": None,
              "n_exam": len(labels), "t": time.time()}
        with self.lock:
            self.st["rounds"].append(rd)
        self.save()
        threading.Thread(target=self.behaviour, args=(gv, k), daemon=True).start()
        worst = max([o["acc"] for o in old], default=None)
        return ("A blind test of %d files (half public, half yours) has been prepared for the reviewer (round %d)." % (len(labels), k) +
                ("" if worst is None else " The reviewer's earlier programs reach at most %.0f%% accuracy on this new sample." % (100 * worst)))

    # ── discriminator requests ───────────────────────────────────────────────
    def score_tell(self, tell_path, labels):
        """Run a tell.py on the current exam files inside the discriminator's jail; returns (accuracy, auc, n_bad)."""
        home = self.td / "disrun"
        jail.prepare_home(home)
        work = home / "work"
        shutil.rmtree(work / "exam", ignore_errors=True)
        (work / "exam").mkdir(parents=True, exist_ok=True)
        shutil.copy(tell_path, work / "tell.py")
        for fn in labels:
            shutil.copy(self.D / "exam" / fn, work / "exam" / fn)

        def one(fn):
            cmd = jail.jail_cmd(home, ["python3", "tell.py", "exam/" + fn], cpus=jail.alloc_cpus("svc/" + self.td.name, cores=2))
            try:
                r = subprocess.run(cmd, env=jail.proot_env(), capture_output=True, text=True, timeout=120)
                return fn, float(r.stdout.strip().split()[-1])
            except Exception:
                return fn, None

        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            out = dict(ex.map(one, sorted(labels)))
        bad = sum(1 for v in out.values() if v is None or not (0.0 <= v <= 1.0))
        pr = {fn: (0.5 if (v is None or not (0.0 <= v <= 1.0)) else v) for fn, v in out.items()}
        correct = sum((1.0 if (p > 0.5) == (labels[fn]["label"] == "P") else 0.0) if p != 0.5 else 0.5 for fn, p in pr.items())
        a = auc([p for fn, p in pr.items() if labels[fn]["label"] == "P"], [p for fn, p in pr.items() if labels[fn]["label"] == "Q"])
        return correct / float(len(labels)), a, bad

    def do_dis_submit(self, note):
        if not self.st["rounds"]:
            return 2, "no exam has been issued yet"
        rd = self.st["rounds"][-1]
        if rd["submits"] >= self.max_submits:
            return 1, "no submissions left in this round"
        if not (self.D / "tell.py").exists() or not (self.D / "REASONS.md").exists():
            return 2, "submit needs tell.py and REASONS.md"
        if len((self.D / "REASONS.md").read_text(errors="ignore").strip()) < 150:
            return 2, ("REASONS.md does not state your decision criteria (measured quantity, values for P and Q, threshold and "
                       "direction for each); a submission without criteria is rejected")
        tmpd = self.priv / "tells" / "_check"
        shutil.rmtree(tmpd, ignore_errors=True)
        tmpd.mkdir()
        shutil.copy(self.D / "tell.py", tmpd / "tell.py")
        ok, why = self.static_check(tmpd, "tell")
        shutil.rmtree(tmpd, ignore_errors=True)
        if not ok:
            return 1, "tell.py rejected: " + why
        k = rd["round"]
        labels = json.loads((self.priv / "exams" / ("r%d.json" % k)).read_text())
        with self.lock:
            rd["submits"] += 1
            j = rd["submits"]
        tp = self.priv / "tells" / ("r%d_s%d.py" % (k, j))
        shutil.copy(self.D / "tell.py", tp)
        shutil.copy(self.D / "REASONS.md", self.priv / "reasons" / ("r%d_s%d.md" % (k, j)))
        acc, a, bad = self.score_tell(tp, labels)
        with self.lock:
            if rd["best_acc"] is None or acc > rd["best_acc"]:
                rd["best_acc"], rd["best_auc"], rd["best_submit"] = acc, a, j
        self.save()
        self.ledger({"type": "dis_submit", "round": k, "submit": j, "acc": acc, "auc": a, "bad_outputs": bad, "note": note})
        self.write_feedback()
        return 0, "exam accuracy %.1f%% on %d files (%d unusable outputs counted as 0.5). Submissions left in this round: %d." % (
            100 * acc, len(labels), bad, self.max_submits - j)

    def write_feedback(self):
        """What the imitator sees: how well the reviewer separated its files, and the reviewer's stated differences."""
        rd = self.st["rounds"][-1]
        L = ["## Reviewer's result on your generator version %d (round %d)" % (rd["gen_version"], rd["round"])]
        if rd["best_acc"] is None:
            L.append("The reviewer has not answered yet.")
        else:
            L.append("A reviewer who saw labelled examples of public files and of your files then classified %d unseen files "
                     "(half of each) with %.0f%% accuracy (50%% = cannot tell them apart; the goal is at most %.0f%%)." % (
                         rd["n_exam"], 100 * rd["best_acc"], 100 * self.thr))
            rp = self.priv / "reasons" / ("r%d_s%d.md" % (rd["round"], rd.get("best_submit", 1)))
            if rp.exists():
                L += ["", "### Differences the reviewer reported (its own words; P = public files, Q = your files)", "",
                      rp.read_text(errors="ignore")[:12000]]
        if rd.get("behaviour"):
            b = rd["behaviour"]
            L += ["", "### Solver behaviour", "A reference solver was run on public files and on your files of the same sizes: " +
                  ("its convergence behaviour is statistically indistinguishable on the two sets." if b["ok"] else
                   "its convergence behaviour DIFFERS between the two sets (%s)." % "; ".join(
                       "%s: public median %.3g vs yours %.3g, p=%.3f" % (s, v["p_med"], v["q_med"], v["p"]) for s, v in b["stats"].items()))]
        (self.G / "feedback").mkdir(parents=True, exist_ok=True)
        (self.G / "feedback" / "latest.md").write_text("\n".join(L) + "\n")

    # ── solver behaviour (no LLM) ────────────────────────────────────────────
    def solver(self):
        b = self.priv / "bin"
        if not (b / "solver").exists():
            env = dict(os.environ)
            env["PATH"] = (self.C.get("toolchain_bin") or os.environ.get("SE_TOOLCHAIN_BIN", "/opt/alphaco/toolchain/bin")) + ":" + env.get("PATH", "")
            r = subprocess.run(["bash", "-c", self.P.BUILD, "build", str(b)], cwd=self.C["solver_tree"], env=env,
                               capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                raise RuntimeError("solver build failed: " + (r.stdout + r.stderr)[-2000:])
        return str(b)

    def run_stats(self, files, tag):
        """Run the reference solver once per file (pinned core, official time limit); anytime-profile statistics."""
        od = self.priv / "behaviour" / tag
        od.mkdir(parents=True, exist_ok=True)
        opts = self.C.get("train_opts", {})
        jobs, meta = [], []
        for name, path in files:
            n = self.P.size(path)
            T = self.P.time_limit(n) * float(self.C.get("time_scale", 1.0))
            sol, log = str(od / (name + ".sol")), str(od / (name + ".log"))
            jobs.append({"cmd": self.P.run_cmd(self.solver(), path, sol, T, 1, opts), "cwd": str(od), "timeout": T + 120, "stdout": log})
            meta.append((name, path, T, sol, log))
        order = sorted(range(len(jobs)), key=lambda i: -meta[i][2])
        res = poolclient.run_jobs(self.C["pool"], [jobs[i] for i in order], prio=int(self.C.get("prio", 7)), tag="twin_" + self.C.get("name", ""))
        out = {}
        for i, r in zip(order, res):
            name, path, T, sol, log = meta[i]
            v = self.P.verify(path, sol, opts)
            tr = self.P.trace(sol, log)
            if not v["ok"] or not tr:
                continue
            final = float(v["cost"])
            q = [c for t, c in tr if t <= 0.25 * T]
            c25 = q[-1] if q else tr[0][1]
            out[name] = {"early_excess_pct": 100.0 * (c25 - final) / final, "last_improvement_frac": min(1.0, tr[-1][0] / T),
                         "cost": final, "n": self.P.size(path)}
        return out

    def behaviour(self, gv, k):
        try:
            pubf = self.priv / "behaviour" / "public.json"
            with self.lock:
                need_pub = not pubf.exists()
            if need_pub:
                pub = self.run_stats([(name, path) for name, path, _ in self.train], "public")
                pubf.write_text(json.dumps(pub))
            pub = json.loads(pubf.read_text())
            snap = self.priv / "gen" / ("v%d" % gv)
            meta = json.loads((snap / "meta.json").read_text())
            names = [n for n in self.split["exam"] if n in meta]
            gf = self.priv / "behaviour" / ("v%d.json" % gv)
            if not gf.exists():
                gen = self.run_stats([(n, str(snap / "sample" / meta[n]["file"])) for n in names], "v%d" % gv)
                gf.write_text(json.dumps(gen))
            gen = json.loads(gf.read_text())
            stats, ok = {}, True
            for s in ("early_excess_pct", "last_improvement_frac"):
                a = [pub[n][s] for n in names if n in pub and n in gen]
                b = [gen[n][s] for n in names if n in pub and n in gen]
                d, p = ks_2samp(a, b)
                med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")   # noqa: E731
                stats[s] = {"ks": d, "p": p, "p_med": med(a), "q_med": med(b), "n": len(a)}
                ok = ok and p >= 0.025
            res = {"ok": ok, "stats": stats, "gen_version": gv}
            with self.lock:
                for rd in self.st["rounds"]:
                    if rd["gen_version"] == gv:
                        rd["behaviour"] = res
            self.save()
            self.ledger({"type": "behaviour", "gen_version": gv, "round": k, **res})
            self.write_feedback()
        except Exception:
            with open(self.td / "logs" / "twin_errors.log", "a") as ef:
                ef.write("%s behaviour v%d\n%s\n" % (time.strftime("%F %T"), gv, traceback.format_exc()))

    # ── certification ────────────────────────────────────────────────────────
    def round_passed(self, rd):
        accs = [rd["best_acc"]] if rd["best_acc"] is not None else []
        accs += [o["acc"] for o in rd.get("old_tells", [])]
        return bool(rd["submits"] > 0 and accs and max(accs) <= self.thr and rd.get("behaviour") and rd["behaviour"]["ok"])

    def certified(self):
        r = self.st["rounds"]
        return len(r) >= 2 and r[-1]["gen_version"] == r[-2]["gen_version"] and self.round_passed(r[-1]) and self.round_passed(r[-2])

    def status(self):
        L = ["generator versions submitted %d, accepted version %d, rounds %d, certified %s" % (
            self.st["gen_version"], self.st["accepted"], self.st["round"], self.certified())]
        for rd in self.st["rounds"]:
            b = rd.get("behaviour")
            L.append("  round %d  gen v%d  discriminator best acc %s (auc %s, %d submits)  old programs max acc %s  behaviour %s  -> %s" % (
                rd["round"], rd["gen_version"], "n/a" if rd["best_acc"] is None else "%.2f" % rd["best_acc"],
                "n/a" if rd["best_auc"] is None else "%.2f" % rd["best_auc"], rd["submits"],
                ("%.2f" % max(o["acc"] for o in rd["old_tells"])) if rd["old_tells"] else "n/a",
                "pending" if not b else ("ok" if b["ok"] else "DIFFERS") + " (p " + ", ".join("%.3f" % v["p"] for v in b["stats"].values()) + ")",
                "pass" if self.round_passed(rd) else "not passed"))
        return "\n".join(L)

    # ── draws for campaigns (phase 1) ────────────────────────────────────────
    def draw(self, k, outdir, prefix, sizes, rng):
        gv = self.st["accepted"]
        snap = self.priv / "gen" / ("v%d" % gv)
        fams = [f["name"] for f in json.loads((snap / "code" / "families.json").read_text())]
        rng.shuffle(fams)
        plan = [("%s-%02d-n%d" % (prefix, i + 1, n), fams[i % len(fams)], n, rng.randrange(1, 2**31 - 1)) for i, n in enumerate(sizes[:k])]
        res = self.gen_run(snap / "code", plan, Path(outdir))
        return {job[0]: {"family": job[1], "n": job[2], "seed": job[3], "ok": res[job[0]][0], "why": res[job[0]][1]} for job in plan}

    # ── request loop ─────────────────────────────────────────────────────────
    def handle(self, who, req):
        cmd, a = req.get("cmd"), req.get("args", [])
        if who == "gen":
            if cmd == "submit":
                return self.do_gen_submit(" ".join(a))
            if cmd == "feedback":
                self.write_feedback() if self.st["rounds"] else None
                p = self.G / "feedback" / "latest.md"
                return 0, p.read_text() if p.exists() else "no feedback yet"
            return 2, "commands: submit \"note\" | feedback"
        if cmd == "submit":
            return self.do_dis_submit(" ".join(a))
        return 2, "commands: submit \"note\""

    def loop(self, who, box):
        (box / "req").mkdir(parents=True, exist_ok=True)
        (box / "res").mkdir(parents=True, exist_ok=True)
        while not (self.td / "STOP").exists():
            for f in sorted((box / "req").glob("*.json")):
                try:
                    req = json.loads(f.read_text())
                except Exception:
                    continue
                t0 = time.time()
                try:
                    rc, txt = self.handle(who, req)
                except Exception:
                    rc, txt = 9, "service error (reported to the operators); try again later\n"
                    with open(self.td / "logs" / "twin_errors.log", "a") as ef:
                        ef.write("%s %s %s\n%s\n" % (time.strftime("%F %T"), who, req, traceback.format_exc()))
                tmp = box / "res" / (f.stem + ".tmp")
                tmp.write_text(json.dumps({"rc": rc, "text": txt}))
                os.replace(tmp, box / "res" / (f.stem + ".json"))
                f.unlink(missing_ok=True)
                with open(self.td / "logs" / "twin.log", "a") as lf:
                    lf.write("%s %s %s rc=%s %.0fs | %s\n" % (time.strftime("%F %T"), who, json.dumps(req)[:160], rc, time.time() - t0,
                                                            txt.split("\n")[0][:200]))
            time.sleep(0.5)

    def serve(self):
        import fcntl
        self._lockf = open(self.priv / "twin.lock", "w")
        try:
            fcntl.flock(self._lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("another twin service is running; exiting")
            return
        for who, box in (("gen", self.G / "box"), ("dis", self.D / "box")):
            threading.Thread(target=self.loop, args=(who, box), daemon=True).start()
        while not (self.td / "STOP").exists():
            for f in (self.td / "cmd").glob("reissue*") if (self.td / "cmd").exists() else []:
                f.unlink()
                self.issue_exam(self.st["accepted"])
                self.ledger({"type": "reissue", "gen_version": self.st["accepted"]})
            time.sleep(2)


if __name__ == "__main__":
    T = Twin(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "serve"
    if mode == "serve":
        T.serve()
    elif mode == "status":
        print(T.status())
