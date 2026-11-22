#!/bin/bash
#SBATCH --job-name=se_arm
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
# One campaign = one job: this job holds the campaign's own evaluation pool (one solver run per physical core), the
# judge, the driver and the agent session (1 physical core set aside for it). Nothing runs on the login node.
#
#   sbatch -J se_F cluster/arm.sbatch cvrp_F4                                    # fixed training set
#   sbatch -J se_T cluster/arm.sbatch cvrp_T4 'rotate={"every_s":3600,"k":25}'   # rotated training set
# Small training set (10 instances, task file tasks/cvrp_blind10.json), fixed and rotated, same initial set:
#   SE_PLAIN=1 SE_TASK=tasks/cvrp_blind10.json sbatch -J se_F cluster/arm.sbatch cvrp_F5
#   SE_PLAIN=1 SE_TASK=tasks/cvrp_blind10.json SE_WAIT_FOR=camp/cvrp_F5/priv/train/initial.json sbatch -J se_T \
#       cluster/arm.sbatch cvrp_T5 train_copy_from=cvrp_F5 'rotate={"every_s":3600,"k":5}'
# Islands (specialists that continue another campaign's line; references to their peers are refreshed every session):
#   SE_PLAIN=1 SE_TASK=tasks/cvrp_isl_M.json SE_START_FROM=cvrp_T5 sbatch -J se_M cluster/arm.sbatch cvrp_M1
#   SE_PLAIN=1 SE_TASK=tasks/cvrp_isl_L.json SE_START_FROM=cvrp_F5 sbatch -J se_L cluster/arm.sbatch cvrp_L1
# Both start from the initial training set of camp/cvrp_T2 (the one used on the first machine). Optional variables:
#   SE_TASK (default tasks/cvrp_blind.json)  SE_BUDGET_H (12)  SE_COPY_FROM (cvrp_T2)  SE_TWIN (twin/cvrp)
# RESUME: submitting the same campaign name again continues it (calibration, committed versions, caches, notes and
# the used budget are on disk; a session cut off with the job is charged). Options after the name are applied to
# the existing campaign (e.g. budget_h=18). Inside a running job: `touch camp/<name>/RESTART` restarts driver and
# judge with the code on disk; `touch camp/<name>/PAUSE` ends the session and the job, to be resumed later.
CAMP=$1; shift
cd "${SE_ROOT:-$HOME/epicco}" || exit 1
. cluster/env.sh
eval "$(python3 cluster/cpus.py ${SE_SESSION_CORES:-1})"      # the QOS allows 32 CPUs per job = 16 physical cores: 15 for the pool, 1 for the session
export SE_POOL="$SE_ROOT/pools/$CAMP"
C=$SE_ROOT/camp/$CAMP
echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname): $SE_NCPU CPUs = $SE_NPHYS physical cores; pool $SE_NPOOL cores, session CPUs $SE_SESSION_CPUS"
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | g++: $(g++ --version | head -1) | python: $(python3 --version)"
[ -s "${SE_CRED:-$HOME/.claude/.credentials.json}" ] || { echo "no credentials file (${SE_CRED:-~/.claude/.credentials.json})"; exit 1; }
echo "project root $SE_ROOT | credentials ${SE_CRED:-$HOME/.claude/.credentials.json}"

mkdir -p "$SE_POOL" logs
rm -f "$SE_POOL/STOP"
python3 fw/pool.py "$SE_POOL" > logs/pool_${CAMP}_$SLURM_JOB_ID.out 2>&1 &
POOL_PID=$!
trap 'touch "$SE_POOL/STOP"; sleep 3; kill $POOL_PID 2>/dev/null' EXIT
sleep 15
cat "$SE_POOL/pool.hb" || { echo "pool did not start"; exit 1; }

cd fw
V4='v4={"base_seeds":[1,2],"cand_seeds":[11],"max_seeds":3,"t_accept":2.33,"t_try":1.0,"t_stop":0.5,"screen_frac":3}'
if [ -n "$SE_WAIT_FOR" ]; then        # e.g. the campaign whose initial training set this one copies is being created by another job
  for i in $(seq 1 120); do [ -e "$SE_ROOT/$SE_WAIT_FOR" ] && break; sleep 5; done
  [ -e "$SE_ROOT/$SE_WAIT_FOR" ] || { echo "$SE_WAIT_FOR did not appear"; exit 1; }
  sleep 5
fi
if [ ! -d "$C" ]; then
  if [ -n "$SE_CLONE_FROM" ]; then    # same initial training set AND the calibration of the original as another campaign
    python3 clone_arm.py "$SE_CLONE_FROM" "$CAMP" "$SE_TASK" "$@" || { echo "clone failed"; exit 1; }
  elif [ -n "$SE_PLAIN" ]; then         # everything comes from the task file and the arguments (no defaults added here)
    python3 setup_camp.py "$CAMP" "$SE_TASK" train_twin="$SE_ROOT/${SE_TWIN:-twin/cvrp}" "$@" || { echo "setup failed"; exit 1; }
  else
    python3 setup_camp.py "$CAMP" "${SE_TASK:-tasks/cvrp_blind.json}" train_twin="$SE_ROOT/${SE_TWIN:-twin/cvrp}"         train_copy_from="${SE_COPY_FROM:-cvrp_T2}" prompt_template=enhance_blind4.md budget_h="${SE_BUDGET_H:-12}"         session_cap_s=10800 "$V4" "$@" || { echo "setup failed"; exit 1; }
  fi
else
  # RESUME of an existing campaign: nothing is set up again; options given on the command line are applied
  echo "resuming $CAMP: $(head -c 160 "$C/priv/drive.json" 2>/dev/null | tr -d '[:space:]')"
  [ -n "$SE_BUDGET_H" ] && set -- "budget_h=$SE_BUDGET_H" "$@"
  [ $# -gt 0 ] && python3 set_opts.py "$C" "$@"
fi
rm -f "$C/STOP"
# the campaign always evaluates on THIS job's pool, however it was created (a campaign made by clone_arm.py outside a
# job had the default pool path: every probe was refused as "pool not reachable", 11-11 22:07-22:22)
python3 set_opts.py "$C" pool="$SE_POOL"
if ! grep -q "mean over seeds" "$C/logs/calib.out" 2>/dev/null; then
  # the original solver on this machine: 3 seeds on every training instance (hardware differs from the first machine)
  python3 judge.py "$C" calib > "$C/logs/calib.out" 2>&1
  tail -4 "$C/logs/calib.out"
  grep -q "mean over seeds" "$C/logs/calib.out" || { echo "calibration failed"; exit 1; }
fi
if [ -n "$SE_START_FROM" ] && [ ! -f "$C/priv/drive.json" ]; then
  # island that continues another campaign's line: its last committed version becomes v1 here (the original stays v0)
  python3 islands.py carry "$CAMP" "$SE_START_FROM" || { echo "carry failed"; exit 1; }
fi
python3 drive.py "$C" >> "$C/logs/drive.out" 2>&1
echo "$(date '+%F %T') driver ended"; tail -3 "$C/logs/drive.log"
