#!/usr/bin/env python3
"""Pinned single-core worker pool. Runs INSIDE a Slurm allocation (sbatch --wrap "python3 pool.py <root>").

Protocol (plain files under <root>):
  queue/<id>.json    {"cmd": [...], "cwd": str|None, "timeout": seconds, "stdout": path, "stderr": path|None, "env": {...}}
  running/<id>.json  the same file while the job runs
  done/<id>.json     {"id", "rc", "elapsed", "timed_out", "cpu", "start"}
Ids sort as priority then submission time, so the queue is FIFO within a priority.
One job per PHYSICAL core (the first hardware thread of every core in this job's cpuset; the sibling stays idle),
so a time-limited solver run always owns a whole core.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def phys_slots():
    seen = {}
    excl = {int(x) for x in os.environ.get("SE_POOL_EXCLUDE", "").replace(",", " ").split()}   # CPUs kept for the agent session
    for c in sorted(os.sched_getaffinity(0) - excl):
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except OSError:
            sib = str(c)
        seen.setdefault(sib, c)
    return sorted(seen.values())


def main(root):
    root = Path(root)
    # SE_POOL_WORKER=<id>: an additional worker on the same spool, running in another Slurm job (more cores for one
    # campaign). It keeps its running jobs in its own directory and its own heartbeat file, never touches the main
    # pool's files, and gives its unfinished jobs back to the queue when it is told to stop (SIGTERM, STOP file).
    wid = os.environ.get("SE_POOL_WORKER", "")
    q, r, d = root / "queue", root / ("running_w" + wid if wid else "running"), root / "done"
    for p in (q, r, d):
        p.mkdir(parents=True, exist_ok=True)
    for f in r.glob("*.json"):            # jobs orphaned by a previous instance of THIS pool/worker: run them again
        os.replace(f, q / f.name)
    hbf = root / ("pool.hb.w" + wid if wid else "pool.hb")
    term = []
    signal.signal(signal.SIGTERM, lambda *a: term.append(1))
    slots = phys_slots()
    free = list(slots)
    live = {}
    tags = {}
    info = {"pid": os.getpid(), "host": os.uname().nodename, "slots": slots, "job": os.environ.get("SLURM_JOB_ID"),
            "started": time.time()}
    (root / ("pool.info.w" + wid if wid else "pool.info")).write_text(json.dumps(info))
    last_hb = 0.0
    while True:
        now = time.time()
        for jid, (proc, job, t0, cpu, fo, fe) in list(live.items()):
            # reap with wait4 so that the CPU time of the run (process + its waited-for children) is recorded:
            # a run that consumed more CPU seconds than wall-clock seconds used more than one core
            el = now - t0
            to = False
            rc, ru = None, None
            try:
                pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
                if pid != 0:
                    rc = -(status & 0x7f) if (status & 0x7f) else (status >> 8)
            except ChildProcessError:
                rc = 0
            if rc is None and el > float(job.get("timeout", 3600)):
                try:
                    os.killpg(proc.pid, 9)
                except OSError:
                    pass
                try:
                    _, _, ru = os.wait4(proc.pid, 0)
                except ChildProcessError:
                    pass
                rc, to = -9, True
            if rc is not None:
                proc.returncode = rc
                job["_cpu"] = (ru.ru_utime + ru.ru_stime) if ru is not None else None
                for fh in (fo, fe):
                    try:
                        fh.close()
                    except Exception:
                        pass
                tmp = d / (jid + ".tmp")
                tmp.write_text(json.dumps({"id": jid, "rc": rc, "elapsed": time.time() - t0, "timed_out": to,
                                           "cpu": cpu, "start": t0, "cpu_s": job.get("_cpu")}))
                os.replace(tmp, d / (jid + ".json"))
                try:
                    (r / (jid + ".json")).unlink()
                except OSError:
                    pass
                free.append(cpu)
                del live[jid]
        if free:
            try:
                names = sorted(n for n in os.listdir(q) if n.endswith(".json"))
            except OSError:
                names = []
            # fair sharing between campaigns that run at the same time: within a priority level the next core goes to
            # the campaign (job "tag") that currently has the fewest running jobs; FIFO inside a campaign
            for nm in names:
                if nm not in tags:
                    try:
                        tags[nm] = json.loads((q / nm).read_text()).get("tag") or ""
                    except Exception:
                        pass
            for nm in list(tags):
                if nm not in names:
                    del tags[nm]
            names = [nm for nm in names if nm in tags]
            while free and names:
                run = {}
                for _, (_, lj, _, _, _, _) in live.items():
                    run[lj.get("tag") or ""] = run.get(lj.get("tag") or "", 0) + 1
                nm = min(names, key=lambda x: (x.split("_")[0], run.get(tags[x], 0), x))
                names.remove(nm)
                try:
                    job = json.loads((q / nm).read_text())
                except Exception:
                    continue
                jid = nm[:-5]
                if jid in live:
                    # this pool is already running that job: its file came back to the queue (a main pool that missed
                    # this worker's heartbeat for two minutes). Starting it again here would overwrite the entry of the
                    # live process, which would then never be reaped and its core never freed (bin packing run, 11-14:
                    # a worker went on with 4 of its 5 cores). Keep the one process and restore its marker.
                    try:
                        os.replace(q / nm, r / nm)
                    except OSError:
                        pass
                    continue
                try:
                    os.replace(q / nm, r / nm)
                except OSError:
                    continue
                cpu = free.pop(0)
                env = dict(os.environ)
                env.update(job.get("env") or {})
                env["OMP_NUM_THREADS"] = "1"
                try:
                    fo = open(job["stdout"], "wb")
                    fe = open(job["stderr"], "wb") if job.get("stderr") else subprocess.STDOUT
                    proc = subprocess.Popen(job["cmd"], cwd=job.get("cwd"), stdout=fo,
                                            stderr=(fe if job.get("stderr") else subprocess.STDOUT),
                                            stdin=subprocess.DEVNULL, env=env, start_new_session=True,
                                            preexec_fn=(lambda c=cpu: os.sched_setaffinity(0, {c})))
                    if not job.get("stderr"):
                        fe = fo
                    live[jid] = (proc, job, time.time(), cpu, fo, fe)
                except Exception as ex:   # bad command: report instead of dying
                    tmp = d / (jid + ".tmp")
                    tmp.write_text(json.dumps({"id": jid, "rc": -1, "elapsed": 0.0, "timed_out": False, "cpu": cpu,
                                               "start": time.time(), "error": repr(ex)}))
                    os.replace(tmp, d / (jid + ".json"))
                    try:
                        (r / nm).unlink()
                    except OSError:
                        pass
                    free.append(cpu)
        if now - last_hb > 10:
            # atomic (11-16): write_text truncates first, and a reader that came in between saw an empty file = "pool not
            # reachable" for the judge, "worker dead" (its runs re-queued) for the main pool
            try:
                hbt = hbf.with_name(hbf.name + ".tmp")
                hbt.write_text(json.dumps({"t": now, "live": len(live), "free": len(free), "queued": len(os.listdir(q))}))
                os.replace(hbt, hbf)
            except OSError:
                pass
            last_hb = now
            if not wid:
                # a worker whose heartbeat stopped (its job died): its unfinished runs go back to the queue
                for wd in root.glob("running_w*"):
                    try:
                        age = now - json.loads((root / ("pool.hb.w" + wd.name[9:])).read_text())["t"]
                    except Exception:
                        age = 1e9
                    if age > 120:
                        for f in wd.glob("*.json"):
                            try:
                                os.replace(f, q / f.name)
                            except OSError:
                                pass
        if (root / "STOP").exists() and not live:
            break
        if term or (wid and (root / "STOP").exists()):
            for jid, (proc, job, t0, cpu, fo, fe) in list(live.items()):     # give the unfinished runs back
                try:
                    os.killpg(proc.pid, 9)
                except OSError:
                    pass
                try:
                    os.replace(r / (jid + ".json"), q / (jid + ".json"))
                except OSError:
                    pass
            try:
                hbf.unlink()
            except OSError:
                pass
            break
        time.sleep(0.1)


if __name__ == "__main__":
    main(sys.argv[1])
