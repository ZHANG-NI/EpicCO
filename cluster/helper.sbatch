#!/bin/bash
#SBATCH --job-name=se_help
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
# Extra evaluation cores for running campaigns:  sbatch cluster/helper.sbatch <camp> [<camp> ...]
# The job's physical cores are split evenly between the named campaigns; for each one a pool worker (fw/pool.py with
# SE_POOL_WORKER) serves that campaign's spool next to its own pool. Same node type as the arms (constraint cpu), so
# a run takes the same time wherever it lands. A worker stops when its campaign's pool stops; unfinished runs go back
# to the queue when this job is cancelled or times out.
cd "${SE_ROOT:-$HOME/alphaco}" || exit 1
. cluster/env.sh
echo "$(date '+%F %T') helper job $SLURM_JOB_ID on $(hostname): $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
N=$#
[ $N -ge 1 ] || { echo "no campaign given"; exit 1; }

# ── self-test on a scratch spool: a main pool and a worker share 40 short jobs, every job must come back exactly once
T=$SE_ROOT/pools/_selftest_$SLURM_JOB_ID; mkdir -p $T
python3 - "$T" <<'PY' || { echo "FAIL self-test: the helper does not attach to any campaign"; rm -rf "$T"; exit 1; }
import json, os, subprocess, sys, time
sys.path.insert(0, "fw")
import poolclient
T = sys.argv[1]
aff = sorted(os.sched_getaffinity(0))
half = aff[:len(aff) // 2]
a = subprocess.Popen([sys.executable, "fw/pool.py", T], env=dict(os.environ, SE_POOL_EXCLUDE=",".join(map(str, aff[len(aff) // 2:]))))
b = subprocess.Popen([sys.executable, "fw/pool.py", T], env=dict(os.environ, SE_POOL_WORKER="t", SE_POOL_EXCLUDE=",".join(map(str, half))))
time.sleep(3)
jobs = [{"cmd": ["bash", "-c", "sleep 1; echo %d" % i], "cwd": T, "timeout": 30, "stdout": "%s/o%d.txt" % (T, i)} for i in range(40)]
t0 = time.time()
res = poolclient.run_jobs(T, jobs, tag="selftest")
ok = all(r["rc"] == 0 for r in res) and all(open("%s/o%d.txt" % (T, i)).read().strip() == str(i) for i in range(40))
cpus = {r["cpu"] for r in res}
both = bool(cpus & set(half)) and bool(cpus - set(half))
open(T + "/STOP", "w").write("x")
a.wait(timeout=30); b.wait(timeout=30)
left = len(os.listdir(T + "/queue")) + len(os.listdir(T + "/running")) + len(os.listdir(T + "/running_wt"))
print("self-test: 40 jobs in %.1fs, all correct %s, both instances worked %s, cores used %d, left in the spool %d" % (time.time() - t0, ok, both, len(cpus), left))
sys.exit(0 if ok and both and left == 0 else 1)
PY
rm -rf "$T"

# ── split the physical cores between the campaigns and start one worker per campaign
PIDS=""
i=0
for CAMP in "$@"; do
  P=$SE_ROOT/pools/$CAMP
  [ -f "$P/pool.hb" ] || { echo "campaign $CAMP has no pool at $P: skipped"; i=$((i+1)); continue; }
  EXCL=$(python3 - $i $N <<'PY'
import os, sys
i, n = int(sys.argv[1]), int(sys.argv[2])
phys = {}
for c in sorted(os.sched_getaffinity(0)):
    try:
        sib = open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c).read().strip()
    except OSError:
        sib = str(c)
    phys.setdefault(sib, []).append(c)
cores = sorted(phys.values())
mine = [core for k, core in enumerate(cores) if k % n == i]
print(",".join(str(c) for core in cores if core not in mine for c in core))
PY
)
  SE_POOL_WORKER="$SLURM_JOB_ID" SE_POOL_EXCLUDE="$EXCL" python3 fw/pool.py "$P" > logs/helper_${CAMP}_$SLURM_JOB_ID.out 2>&1 &
  PIDS="$PIDS $!"
  echo "worker for $CAMP: pid $! (excluded CPUs: $EXCL)"
  i=$((i+1))
done
trap 'kill -TERM $PIDS 2>/dev/null; sleep 8' TERM INT EXIT
sleep 20
for CAMP in "$@"; do echo "$CAMP worker heartbeat: $(cat $SE_ROOT/pools/$CAMP/pool.hb.w$SLURM_JOB_ID 2>/dev/null)"; done
wait $PIDS
echo "$(date '+%F %T') all workers ended"
