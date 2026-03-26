#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --partition=compsci-gpu
# Note: Removed static #SBATCH --output here to handle it dynamically below

# 1. Generate today's date
TODAY=$(date +%Y-%m-%d)

# 2. Define the directory path
# We use SLURM_JOB_ID (or ARRAY_JOB_ID) to keep specific runs separated
LOGDIR="slurm/logs/${TODAY}/${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

# 3. Redirect all output and errors to that folder
exec > >(tee -a "${LOGDIR}/log.out") 2> >(tee -a "${LOGDIR}/log.err")

# ----------------------------
# CONFIG & RUN
# ----------------------------
FILE=$1

python -m $FILE --base-config "mc" --config "{\"TOTAL_TIMESTEPS\": 1000000 }"