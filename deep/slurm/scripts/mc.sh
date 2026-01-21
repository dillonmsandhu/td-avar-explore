#!/bin/bash
#SBATCH --job-name=mc_sweep
#SBATCH --output=slurm/logs/%j/log.out
#SBATCH --error=slurm/logs/%j/log.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

LOGDIR=slurm/logs/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

# ----------------------------
# CONFIG
# ----------------------------

FILE=$1
# EFFECTIVE_VISITS_TO_REMAIN_OPT=$2
DATE=$(date +%Y-%m-%d)

# ----------------------------
# RUN
# ----------------------------

echo "Running $FILE with MC"

python $FILE --base-config "mc" --run_suffix "${DATE}/LESS_WHITEN"