"""Client side of pool.py: submit a list of jobs, wait for all of them, return their results in order."""
import json
import os
import random
import time
from pathlib import Path


def run_jobs(pool_root, jobs, prio=5, poll=0.3, on_wait=None, tag=None):
    root = Path(pool_root)
    q, d = root / "queue", root / "done"
    ids = []
    for i, job in enumerate(jobs):
        jid = "%d_%d_%05d_%04x" % (prio, time.time_ns(), i, random.getrandbits(16))
        tmp = q / (jid + ".tmp")
        if tag and "tag" not in job:
            job = dict(job, tag=tag)
        tmp.write_text(json.dumps(job))
        os.replace(tmp, q / (jid + ".json"))
        ids.append(jid)
    res = [None] * len(ids)
    left = set(range(len(ids)))
    t0 = time.time()
    while left:
        for i in list(left):
            f = d / (ids[i] + ".json")
            if f.exists():
                try:
                    res[i] = json.loads(f.read_text())
                except Exception:
                    continue
                left.discard(i)
                try:
                    f.unlink()
                except OSError:
                    pass
        if left:
            if on_wait:
                on_wait(len(left), time.time() - t0)
            time.sleep(poll)
    return res


def pool_alive(pool_root, max_age=60, tries=4, wait=0.5):
    """True when the pool's heartbeat is younger than max_age. A heartbeat that cannot be read (missing, empty, half
    written, file system hiccup) is read again a few times before the pool is called unreachable (11-16: one unreadable
    read refused an evaluation of a healthy pool)."""
    for k in range(max(1, tries)):
        try:
            hb = json.loads((Path(pool_root) / "pool.hb").read_text())
            return time.time() - hb["t"] < max_age
        except Exception:
            if k + 1 < tries:
                time.sleep(wait)
    return False
