#!/bin/bash
# launch_arm.sh <campaign> <task.json> [setup_camp key=value ...]
# Create a campaign, calibrate the original solver on its training set, then run the campaign driver.
# If the file $SE/logs/WAIT_POOL_RESTART exists, first wait until restart_pool.sh has logged a restart newer than it.
SE=${SE_ROOT:-/opt/epicco}; cd $SE/framework || exit 1
CAMP=$1; TASK=$2; shift 2
if [ -f $SE/logs/WAIT_POOL_RESTART ]; then
  while [ ! $SE/logs/pool_restart.log -nt $SE/logs/WAIT_POOL_RESTART ]; do sleep 10; done
  sleep 25
fi
python3 setup_camp.py "$CAMP" "$TASK" "$@" > $SE/logs/setup_$CAMP.out 2>&1 || { echo "setup failed" >> $SE/logs/setup_$CAMP.out; exit 1; }
C=$SE/camp/$CAMP
python3 judge.py $C calib > $C/logs/calib.out 2>&1
if grep -q "mean over seeds" $C/logs/calib.out; then
  exec python3 drive.py $C > $C/logs/drive.out 2>&1 < /dev/null
fi
echo "calibration failed" >> $C/logs/drive.out
