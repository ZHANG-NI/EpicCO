#!/usr/bin/env python3
"""drive.py <camp_dir>: one campaign = the judge + the enhancer sessions (+ the instance-agent rounds), all jailed.

The enhancer gets `budget_h` hours of session time in total (sessions are restarted with a fresh context until the
budget is used; the cut-off is enforced here and never mentioned to the model). Time lost to service limits or to a
session that dies at once is not charged.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

FW = Path(__file__).resolve().parent
sys.path.insert(0, str(FW))
import jail          # noqa: E402

DENY = ("Monitor,ScheduleWakeup,TaskOutput,TaskStop,Agent,Task,Workflow,CronCreate,CronDelete,CronList,WebSearch,"
        "WebFetch,PushNotification,RemoteTrigger,SendMessage,EnterWorktree,ExitWorktree,Artifact,NotebookEdit,Skill")
QUOTA = re.compile(r"(session limit|rate limit|usage limit|weekly limit|hit your .{0,20}limit|overloaded|limit reached|resets? at)", re.I)
AUTH = re.compile(r"(authentication_error|invalid api key|oauth token|please run /login|401)", re.I)


def log(cd, msg):
    with open(cd / "logs" / "drive.log", "a") as f:
        f.write("%s %s\n" % (time.strftime("%F %T"), msg))


def ask(box, cmd, args=(), timeout=7200):
    rid = "drv_%d" % time.time_ns()
    tmp = box / "req" / (rid + ".tmp")
    tmp.write_text(json.dumps({"cmd": cmd, "args": list(args)}))
    os.replace(tmp, box / "req" / (rid + ".json"))
    res = box / "res" / (rid + ".json")
    t0 = time.time()
    while not res.exists():
        if time.time() - t0 > timeout:
            return "(no answer)"
        time.sleep(0.5)
    d = json.loads(res.read_text())
    res.unlink()
    return d.get("text", "")


def run_session(cd, who, prompt, timeout, tag, model, tick=None):
    """tick(elapsed) is called every few seconds while the session runs; when it returns a string the session is
    ended at once and that string is returned as r["interrupt"]."""
    logs = cd / "logs"
    pf, tf, ef = logs / (tag + ".prompt.md"), logs / (tag + ".jsonl"), logs / (tag + ".err")
    pf.write_text(prompt)
    env = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1",
           "BASH_DEFAULT_TIMEOUT_MS": "1800000", "BASH_MAX_TIMEOUT_MS": "3600000",
           # second layer of the network block: every child process inherits a dead proxy; only the model API hosts bypass it
           "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
           "http_proxy": "http://127.0.0.1:9", "NO_PROXY": ",".join(jail.ALLOW_HOSTS), "no_proxy": ",".join(jail.ALLOW_HOSTS)}
    marker = "%s:%s" % (cd.name, tag)
    env["SE_SESSION"] = marker
    cpus = jail.alloc_cpus("%s/%s" % (cd.name, who), cores=2)
    cmd = jail.jail_cmd(cd / who, ["claude", "-p", "--model", model, "--dangerously-skip-permissions",
                                   "--disallowedTools", DENY, "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                                   "--output-format", "stream-json", "--verbose", "--max-turns", "5000"], env=env, cred=True, cpus=cpus)
    t0 = time.time()
    with open(pf, "rb") as fin, open(tf, "wb") as fout, open(ef, "wb") as ferr:
        p = subprocess.Popen(cmd, stdin=fin, stdout=fout, stderr=ferr, env=jail.proot_env(), start_new_session=True)
        (logs / (tag + ".pid")).write_text(str(p.pid))
        timed, interrupt = False, None
        while True:
            try:
                p.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
            el = time.time() - t0
            if tick:
                interrupt = tick(el)
            if el >= max(1, timeout):
                timed = True
            if timed or interrupt:
                try:
                    os.killpg(p.pid, 9)
                except OSError:
                    pass
                p.wait()
                break
    el = time.time() - t0
    jail.kill_session(marker)            # whatever the session started and left behind
    tail = b""
    try:
        with open(tf, "rb") as f:
            f.seek(max(0, os.path.getsize(tf) - 20000))
            tail = f.read()
    except OSError:
        pass
    tail_s = tail.decode("utf-8", "ignore") + open(ef, "rb").read()[-4000:].decode("utf-8", "ignore")
    final = ""
    for line in tail_s.splitlines()[::-1]:
        if '"type":"result"' in line:
            try:
                final = json.loads(line).get("result", "") or ""
            except Exception:
                pass
            break
    quota = (not timed) and bool(QUOTA.search(final or tail_s[-3000:])) and (p.returncode != 0 or el < 600 or "is_error\":true" in tail_s)
    auth = (not timed) and bool(AUTH.search(tail_s[-3000:])) and el < 300
    if interrupt:
        quota = auth = False
    return {"elapsed": el, "timed": timed, "rc": p.returncode, "quota": quota, "auth": auth, "final": final,
            "interrupt": interrupt}


def fill(tpl, **kw):
    class D(dict):
        def __missing__(self, k):
            return ""
    return tpl.format_map(D(kw))


def enh_prompt(cd, C, n):
    pr = C["prompt"]
    fresh = bool(C.get("fresh"))
    kw = dict(pr)
    kw.update(session=n, cwd=jail.REAL_HOME + "/work", n_train=C["n_train"],
              status=ask(cd / "enh" / "work" / "box", "status"),
              fresh_text=("To keep it that way, new instances of the same problem class are delivered to `instances/fresh/` "
                          "while you work (announced in `NOTICES.md`); they have no best-known values, so their gap is "
                          "measured against the best cost any version has reached on them.\n") if fresh else "",
              dev_fresh=" plus half of the current fresh instances" if fresh else "",
              full_fresh=" plus the current fresh instances" if fresh else "",
              gate_text=("the mean gap on the public benchmark must improve on the committed version by more than the "
                         "run-to-run noise, with no failed run, and the number of instances at their best-known value must not "
                         "drop by more than one below the best committed version" +
                         ("; in addition the mean gap on the fresh instances must not get worse than the committed version's "
                          "nor than the original solver's." if fresh else ".")))
    rt = C.get("retest")
    if C.get("v4") and C["v4"].get("rule") == "decrease2":
        kw["blind_gate_text"] = (
            ("this tree and the committed version are run on the same seeds, and per instance their mean costs are "
             "compared (after every commit a new set of seeds is used). A " if C["v4"].get("paired") else
             "per instance, the mean cost of this tree over its seeds is compared with the committed version's baseline. A ") +
            "tree passes at once when the first round of seeds shows an improvement of at least %s standard errors over the "
            "training set. When the first round shows a lower mean cost but not that clearly, the service measures a second "
            "round with new seeds by itself, and the tree passes when the mean cost is lower in that round too (%d rounds in "
            "a row)%s. In both cases: no failed run, and not worse than the original solver. A small real gain therefore "
            "passes, and a single lucky round does not." % (
                C["v4"].get("t_accept", 1.7), int(C["v4"].get("rounds", 2)),
                (" and the improvement over the seeds of all rounds together is at least %s standard errors" % C["v4"]["t_second"])
                if float(C["v4"].get("t_second", 0) or 0) > 0 else ""))
        kw["retest_time_text"] = ""
    elif C.get("v4"):
        V = C["v4"]
        kw["blind_gate_text"] = (
            "per instance, the mean cost of this tree over its seeds is compared with the mean cost of the committed "
            "version's baseline runs, and the mean relative difference over the training set must be an improvement of at "
            "least %s standard errors (re-measured on up to %d seeds when close), with no failed run and not worse than "
            "the original solver." % (V.get("t_accept", 2.33), int(V.get("max_seeds", 3))))
        kw["retest_time_text"] = ""
    elif rt:
        kw["blind_gate_text"] = (
            "this tree and the committed version are compared instance by instance on the current training set (same "
            "instances, same seeds), and the mean relative cost difference must be an improvement of at least %s standard "
            "errors, with no failed run and a mean gap not worse than the original solver's. A tree that is close to that "
            "threshold after the first seed is measured with more seeds automatically (up to %d in total), so a small but "
            "consistent improvement can pass and a lucky single measurement cannot." % (
                rt.get("t_accept", 2.33), int(rt.get("max_seeds", 3))))
        kw["retest_time_text"] = " (longer when the service decides to re-measure with more seeds)"
    else:
        kw["blind_gate_text"] = ("the mean gap on the current training set must improve on the committed version by more "
                                 "than the run-to-run noise, with no failed run, and must not be worse than the original "
                                 "solver's.")
        kw["retest_time_text"] = ""
    kw["rotate_text"] = ("\nThe training set is refreshed from time to time: some instances are replaced by new ones of the same "
                         "kind (announced in `NOTICES.md`). Measurements and the commit gate always use the current set, so do "
                         "not rely on results obtained on instances that are no longer there." if C.get("rotate") else "")
    W_ = cd / "enh" / "work"
    briefs = sorted(p.name for p in W_.glob("REF_*.md"))
    kw["ref_text"] = ""
    if briefs:
        kw["ref_text"] += "\nBriefs of the other projects, kept up to date by the service: %s." % ", ".join("`%s`" % b for b in briefs)
    if (W_ / "offers").is_dir():
        kw["ref_text"] += (" `offers/` holds the changes the scout projects have committed, one directory each "
                           "(`change.diff`, `note.md`); new ones are announced in `NOTICES.md`.")
    kw["island_text"] = ("\n" + C["island_text"]) if C.get("island_text") else ""
    return fill((FW / "prompts" / C.get("prompt_template", "enhance.md")).read_text(), **kw)


def gen_prompt(cd, C, n):
    kw = dict(C["prompt"])
    kw.update(session=n, cwd=jail.REAL_HOME + "/work", n_train=C["n_train"])
    return fill((FW / "prompts" / "instances.md").read_text(), **kw)


def main(camp_dir):
    cd = Path(camp_dir).resolve()
    C = json.loads((cd / "camp.json").read_text())
    (cd / "logs").mkdir(exist_ok=True)
    model = C.get("model", "claude-opus-5-5")
    budget = float(C.get("budget_h", 4.0)) * 3600
    sp = cd / "priv" / "drive.json"
    st = json.loads(sp.read_text()) if sp.exists() else {"used": 0.0, "enh": 0, "gen": 0, "done_streak": 0, "started": time.time()}
    lock = threading.Lock()
    import fcntl
    global _DRIVE_LOCK                       # exactly one driver per campaign (a second one would double every session)
    _DRIVE_LOCK = open(cd / "priv" / "drive.lock", "w")
    try:
        fcntl.flock(_DRIVE_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another driver is running this campaign; exiting")
        sys.exit(4)

    def save():
        with lock:
            sp.write_text(json.dumps(st, indent=1))

    # RESUME. A driver that died with its job (node failure, time limit, cancel) left: the elapsed time of the session
    # that was running (recorded every 30 s as "inflight") -> charged now; requests whose client is dead -> removed,
    # so that the judge does not spend an evaluation on an answer nobody reads; stale control files -> removed.
    if st.get("inflight"):
        st["used"] += st["inflight"]
        log(cd, "resume: the session that was running when the driver died is charged %.0fs" % st["inflight"])
        st["inflight"] = 0.0
        st["resumes"] = st.get("resumes", 0) + 1
    st.pop("finished", None)
    save()
    for who_ in ("enh", "gen"):
        for sub_ in ("req", "res"):
            for f in (cd / who_ / "work" / "box" / sub_).glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
    for f in ("RESTART", "PAUSE", "DONE", "PAUSED"):
        (cd / f).unlink(missing_ok=True)
    purged = 0
    for f in (Path(C["pool"]) / "queue").glob("*.json"):       # runs queued by the judge that died with the driver
        try:
            if json.loads(f.read_text()).get("tag") == C.get("name"):
                f.unlink()
                purged += 1
        except (OSError, ValueError):
            pass
    if purged:
        log(cd, "resume: %d queued runs of the previous judge removed from the pool" % purged)
    last_beat = [0.0]

    def tick(el):
        if time.time() - last_beat[0] > 30:
            st["inflight"] = el
            save()
            last_beat[0] = time.time()
        for f in ("STOP", "PAUSE", "RESTART"):
            if (cd / f).exists():
                return f
        return None

    jail.write_net_files(Path(jail.SE) / "common" / "etc")
    judge = subprocess.Popen([sys.executable, str(FW / "judge.py"), str(cd), "serve"],
                             stdout=open(cd / "logs" / "judge.out", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
    (cd / "logs" / "judge.pid").write_text(str(judge.pid))
    log(cd, "driver start pid %d judge %d model %s budget %.0fs used %.0fs fresh=%s" % (os.getpid(), judge.pid, model, budget, st["used"], C.get("fresh")))
    stop = threading.Event()

    def gen_loop():
        gap, long_gap = float(C.get("gen_gap_s", 900)), float(C.get("gen_long_gap_s", 1800))
        last_end, last_ver = 0.0, -1
        while not stop.is_set():
            try:
                ver = json.loads((cd / "priv" / "state.json").read_text()).get("version", 0)
            except Exception:
                ver = 0
            now = time.time()
            due = st["gen"] == 0 or (now - last_end >= gap and (ver != last_ver or now - last_end >= long_gap))
            if not due:
                time.sleep(15)
                continue
            st["gen"] += 1
            save()
            n = st["gen"]
            log(cd, "gen round %d start (committed v%d)" % (n, ver))
            r = run_session(cd, "gen", gen_prompt(cd, C, n), float(C.get("gen_timeout_s", 2400)), "gen_%03d" % n, model)
            log(cd, "gen round %d end %.0fs rc=%s timed=%s quota=%s" % (n, r["elapsed"], r["rc"], r["timed"], r["quota"]))
            last_end, last_ver = time.time(), ver
            if r["quota"] or r["auth"] or (r["elapsed"] < 120 and r["rc"] != 0):
                st["gen"] -= 1
                save()
                stop.wait(300)

    gt = None
    if C.get("fresh") and not C.get("twin"):      # architecture v2: the judge draws from the certified twin instead
        gt = threading.Thread(target=gen_loop, daemon=True)
        gt.start()

    def migrate_loop():                     # event-driven migration: a peer's new commit reaches this island within minutes
        import islands
        while not stop.is_set():
            stop.wait(float(C.get("migrate_poll_s", 180)))
            if stop.is_set():
                break
            try:
                got = islands.refresh_refs(cd.name)
                if got:
                    log(cd, "migration: references updated %s" % ", ".join("%s v%s" % g for g in got))
            except Exception as ex:
                log(cd, "migration failed (campaign continues): %r" % (ex,))

    if C.get("peers"):
        threading.Thread(target=migrate_loop, daemon=True).start()

    fails = 0
    why = "budget used"
    while st["used"] < budget - 60:
        ctl = [f for f in ("STOP", "PAUSE", "RESTART") if (cd / f).exists()]
        if ctl:
            why = ctl[0]
            log(cd, "%s file: leaving" % why)
            break
        if judge.poll() is not None:
            judge = subprocess.Popen([sys.executable, str(FW / "judge.py"), str(cd), "serve"],
                                     stdout=open(cd / "logs" / "judge.out", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
            (cd / "logs" / "judge.pid").write_text(str(judge.pid))
            log(cd, "judge restarted pid %d" % judge.pid)
        if C.get("sync_main") and C.get("role") in ("secondary", "specialist") and st["enh"] >= 1:
            # session boundary of a secondary island: its committed version follows the main island's (judge.apply_sync;
            # the judge owns the campaign state, so the driver only asks and waits for the answer)
            try:
                rid = "%d_%d" % (st["enh"] + 1, int(time.time()))
                rp = cd / "priv" / "sync_result.json"
                (cd / "SYNC").write_text(json.dumps({"id": rid}))
                t_ask, ans = time.time(), None
                while time.time() - t_ask < float(C.get("sync_wait_s", 600)) + 120:
                    try:
                        a = json.loads(rp.read_text())
                        if a.get("id") == rid:
                            ans = a
                            break
                    except (OSError, ValueError):
                        pass
                    if judge.poll() is not None:
                        break
                    time.sleep(2)
                (cd / "SYNC").unlink(missing_ok=True)
                if ans is None:
                    log(cd, "sync: no answer from the judge (campaign continues on its own version)")
                elif ans.get("done"):
                    log(cd, "sync: committed version set to the main island's v%s (here v%s, seed block %s, %s runs copied)" % (
                        ans.get("main_version"), ans.get("version"), ans.get("block"), ans.get("runs_copied")))
                else:
                    log(cd, "sync: not done (%s)" % (ans.get("skipped") or ans.get("error") or "?").replace("\n", " ")[:200])
            except Exception as ex:
                log(cd, "sync failed (campaign continues): %r" % (ex,))
        if C.get("peers"):                      # migration: the peers' committed versions, up to date, before the session
            try:
                import islands
                got = islands.refresh_refs(cd.name)
                if got:
                    log(cd, "migration: references updated %s" % ", ".join("%s v%s" % g for g in got))
            except Exception as ex:
                log(cd, "migration failed (campaign continues): %r" % (ex,))
        st["enh"] += 1
        n = st["enh"]
        save()
        log(cd, "enh session %d start, %.0fs of budget left" % (n, budget - st["used"]))
        # session_cap_s: a session is ended after that long and the next one starts with an empty context and NOTES.md
        # (11-10: one 4-hour session per arm stayed on its first idea's parameters for 3.5 hours)
        r = run_session(cd, "enh", enh_prompt(cd, C, n), min(budget - st["used"], float(C.get("session_cap_s", 1e12))),
                        "enh_%03d" % n, model, tick=tick)
        charged = r["elapsed"]
        st["inflight"] = 0.0
        if r.get("interrupt"):                   # ended by a control file: charge the time, leave through the loop head
            st["used"] += charged
            save()
            log(cd, "enh session %d interrupted by %s after %.0fs, used %.0fs" % (n, r["interrupt"], r["elapsed"], st["used"]))
            continue
        if r["quota"] or r["auth"] or (r["elapsed"] < 120 and r["rc"] != 0):
            charged = 0.0 if r["elapsed"] < 600 else r["elapsed"]
            fails += 1
            log(cd, "enh session %d ended abnormally after %.0fs (rc=%s quota=%s auth=%s): %s" % (
                n, r["elapsed"], r["rc"], r["quota"], r["auth"], r["final"][-300:].replace("\n", " ")))
            st["used"] += charged
            save()
            if fails >= 40:
                log(cd, "GIVING UP: 40 abnormal sessions")
                break
            time.sleep(300)
            continue
        fails = 0
        st["used"] += charged
        m = re.search(r'"status"\s*:\s*"(continue|done)"', r["final"])
        st["done_streak"] = st["done_streak"] + 1 if (m and m.group(1) == "done") else 0
        save()
        log(cd, "enh session %d end %.0fs timed=%s rc=%s status=%s used %.0fs" % (
            n, r["elapsed"], r["timed"], r["rc"], m.group(1) if m else "?", st["used"]))
        if st["done_streak"] >= 2:
            why = "agent done"
            log(cd, "agent reported done twice: ending early")
            break
    stop.set()
    finished = why in ("budget used", "agent done")
    if finished:
        st["finished"] = time.time()
    save()
    # stop a running instance-agent session, then the judge
    for pf in (cd / "logs").glob("gen_*.pid"):
        try:
            os.killpg(int(pf.read_text()), 9)
        except (OSError, ValueError):
            pass
    time.sleep(2)
    # the judge leaves when it sees STOP; for a pause or a restart the file is only there for a moment
    made_stop = not (cd / "STOP").exists()
    if made_stop:
        (cd / "STOP").write_text(why + "\n")
    time.sleep(5)
    try:
        judge.terminate()
        judge.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if finished:
        (cd / "DONE").write_text(json.dumps(st))
        log(cd, "CAMPAIGN_DONE used %.0fs enh sessions %d gen rounds %d" % (st["used"], st["enh"], st["gen"]))
        return
    # not finished: everything is on disk (budget used, committed versions, caches, notes); starting the driver again
    # continues the campaign. RESTART = do that right now, in this job, with the code that is on disk now.
    if made_stop:
        (cd / "STOP").unlink(missing_ok=True)
    if why == "RESTART":
        (cd / "RESTART").unlink(missing_ok=True)
        log(cd, "RESTART: re-executing the driver (used %.0fs of %.0fs)" % (st["used"], budget))
        os.execv(sys.executable, [sys.executable] + sys.argv)
    (cd / "PAUSED").write_text(json.dumps(st))
    log(cd, "CAMPAIGN_PAUSED (%s) used %.0fs of %.0fs; start the driver again to continue" % (why, st["used"], budget))


if __name__ == "__main__":
    # framework/HOLD: do not start yet (a framework update is being installed); when the file is removed, re-execute this
    # script so that the code now on disk is the code that runs
    if (FW / "HOLD").exists():
        while (FW / "HOLD").exists():
            time.sleep(5)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    # <camp>/SWITCH (content: another campaign's directory): this job leaves this campaign paused and continues with the
    # other one, on the same pool (the job slot is kept). Use: write SWITCH, then `touch RESTART`.
    _sw = Path(sys.argv[1]) / "SWITCH"
    if _sw.exists():
        _nxt = _sw.read_text().strip()
        _sw.unlink()
        if _nxt and (Path(_nxt) / "camp.json").exists():
            _cd = Path(sys.argv[1]).resolve()
            try:
                (_cd / "PAUSED").write_text((_cd / "priv" / "drive.json").read_text())
            except OSError:
                pass
            _n = 0                    # the paused campaign's queued runs must not occupy the pool of the next one
            try:
                _C = json.loads((_cd / "camp.json").read_text())
                for _f in (Path(_C["pool"]) / "queue").glob("*.json"):
                    try:
                        if json.loads(_f.read_text()).get("tag") == _C.get("name"):
                            _f.unlink()
                            _n += 1
                    except (OSError, ValueError):
                        pass
            except (OSError, ValueError, KeyError):
                pass
            log(_cd, "CAMPAIGN_PAUSED (SWITCH): %d queued runs removed; this job continues with %s" % (_n, _nxt))
            os.execv(sys.executable, [sys.executable, sys.argv[0], str(Path(_nxt).resolve())])
    main(sys.argv[1])
