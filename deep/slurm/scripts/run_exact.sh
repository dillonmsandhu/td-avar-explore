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
SUFFIX=$1

# python run_exact.py --script algos/3_26_true_val.py --suffix ${SUFFIX} --config '{"BONUS_SCALE": 1.0, "EPISODIC": true, "ABSORBING_TERMINAL_STATE": false}'
python run_exact.py --script algos/3_30_true_val_beta_decay.py --suffix ${SUFFIX} --config '{"BONUS_SCALE": 50.0, "EPISODIC": true, "ABSORBING_TERMINAL_STATE": true, "MIN_COV_LR": 0.01}'
