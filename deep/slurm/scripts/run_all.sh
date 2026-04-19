#!/bin/bash
#SBATCH --job-name=run-all
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --partition=compsci-gpu

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

python run_all.py --script algos/${FILE} --suffix ${2} --config '{"GAE_LAMBDA_i": 0.6, "NUM_STEPS": 256, "NUM_ENVS": 32}'
# VAL_LOSS_WEIGHT
# RND_LOSS_WEIGHT
# INV_LOSS_WEIGHT