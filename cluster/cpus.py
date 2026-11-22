#!/usr/bin/env python3
"""cpus.py [session_cores=2]: split this job's CPUs into the evaluation pool and the agent session.
Prints two shell assignments: SE_SESSION_CPUS (all hardware threads of the last `session_cores` physical cores of the
job) and SE_POOL_EXCLUDE (the same list); the pool then runs one solver per remaining physical core."""
import os
import sys

k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
aff = sorted(os.sched_getaffinity(0))
phys = {}
for c in aff:
    try:
        sib = open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c).read().strip()
    except OSError:
        sib = str(c)
    phys.setdefault(sib, []).append(c)
cores = sorted(phys.values())
sess = [c for core in cores[-k:] for c in core]
lst = ",".join(str(c) for c in sess)
print('export SE_SESSION_CPUS="%s"; export SE_POOL_EXCLUDE="%s"; SE_NPHYS=%d; SE_NPOOL=%d; SE_NCPU=%d' % (
    lst, lst, len(cores), len(cores) - k, len(aff)))
