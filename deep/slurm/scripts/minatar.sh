#!/bin/bash
#SBATCH --job-name=minatar_sweep
#SBATCH --output=slurm/logs/%A/%a.out
#SBATCH --error=slurm/logs/%A/%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --array=0-3

LOGDIR=slurm/logs/${SLURM_ARRAY_JOB_ID}
mkdir -p "$LOGDIR"

# ----------------------------
# CONFIG
# ----------------------------

FILE=$1

DATE=$(date +%Y-%m-%d)
ENVS=("Breakout-MinAtar" "Asterix-MinAtar" "SpaceInvaders-MinAtar" "Freeway-MinAtar")
ENV=${ENVS[$SLURM_ARRAY_TASK_ID]}

# ----------------------------
# RUN
# ----------------------------

echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running $FILE on $ENV"

python $FILE \
    --config "{\"ENV_NAME\": \"$ENV\"}" \
    --base-config "min" \
    --run_suffix "${DATE}/LESS_WHITEN"