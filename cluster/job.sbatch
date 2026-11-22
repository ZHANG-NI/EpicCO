#!/bin/bash
#SBATCH --job-name=se_job
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
# job.sbatch <script> [args ...]: run one of our own scripts (data preparation, packing, tables over large files) in
# a job instead of on the login node. No pool, no solver session.
#   sbatch -J se_prep --mem=200G cluster/job.sbatch cluster/mwis_prep.sh
cd "${SE_ROOT:-$HOME/alphaco}" || exit 1
. cluster/env.sh
echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname): $*"
bash "$@"
echo "JOB_DONE rc=$? $(date '+%F %T')"
