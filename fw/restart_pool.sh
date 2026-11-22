#!/bin/bash
# restart_pool.sh [max_wait_s]: restart the pinned evaluation pool with the currently deployed pool.py at a moment
# when it is idle (nothing running, nothing queued). The Slurm job is cancelled by its numeric id only. Jobs that were
# running at the cancel (none, if idle) are re-queued by the new pool instance.
SE=${SE_ROOT:-/opt/epicco}; MAXW=${1:-3600}; T0=$(date +%s)
while true; do
  IDLE=$(python3 -c "import json,time;h=json.load(open('$SE/pool/pool.hb'));print(int(h['live']==0 and h['queued']==0 and time.time()-h['t']<30))" 2>/dev/null)
  [ "$IDLE" = "1" ] && [ -z "$(ls $SE/pool/queue $SE/pool/running 2>/dev/null | grep json)" ] && break
  [ $(( $(date +%s) - T0 )) -gt $MAXW ] && { echo "$(date '+%F %T') pool never idle in ${MAXW}s: not restarted" >> $SE/logs/pool_restart.log; exit 1; }
  sleep 3
done
JOB=$(python3 -c "import json;print(json.load(open('$SE/pool/pool.info'))['job'])")
scancel "$JOB"
sleep 5
cd $SE && NEW=$(sbatch --parsable -p main -c 64 --mem=64G -t 7-00:00:00 -J se_pool -o $SE/logs/pool_%j.out --wrap "python3 $SE/fw/pool.py $SE/pool")
echo "$(date '+%F %T') pool restarted: old job $JOB, new job $NEW" >> $SE/logs/pool_restart.log
