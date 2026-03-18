#!/bin/bash
#SBATCH --job-name=run-debug
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
SUFFIX=$2

# Basic PPO
# python run_debug.py --script algos/${FILE} --suffix "${SUFFIX}" --config '{"BONUS_SCALE": 0.0, "LSTD_PRIOR_SAMPLES": 0.0}'

# # Intrinsic Reward
# python run_debug.py --script algos/${FILE} --suffix "${SUFFIX}" --config '{"BONUS_SCALE": 1.0, "LSTD_PRIOR_SAMPLES": 0.0}'

# # # Faster decay of intrinsic bonus vector (b):
# python run_debug.py --script algos/${FILE} --suffix "${SUFFIX}" --config '{"BONUS_SCALE": 1.0, "MIN_LSTD_LR_RI": 0.1, "LSTD_PRIOR_SAMPLES": 0.0}'

# # continuing
python run_debug.py --script algos/${FILE} --suffix "${SUFFIX}" --config '{"BONUS_SCALE": 1.0, "LSTD_PRIOR_SAMPLES": 50.0, 
"EPISODIC": true}'

# # Intrinsic Reward plus optimistic initialization
# python run_debug.py --script algos/${FILE} --suffix "${SUFFIX}" --config '{"BONUS_SCALE": 1.0, "LSTD_PRIOR_SAMPLES": 10.0}'
