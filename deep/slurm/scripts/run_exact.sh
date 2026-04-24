#!/bin/bash
#SBATCH --job-name=run-exact
#SBATCH --time=1:00:00
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

# python run_exact.py --script algos/3_26_true_val.py --suffix ${SUFFIX} --config '{"BONUS_SCALE": 1.0, "EPISODIC": true, "ABSORBING_GOAL_STATE": false}'
python run_exact.py --script algos/${FILE} --suffix ${SUFFIX} --config '{"TOTAL_TIMESTEPS": 1000000, "N_SEEDS": 2, "CALC_TRUE_VALUES": true, "SCHEDULE_BETA": true, "GAMMA_i": 0.9}'


#  --config '{"BONUS_SCALE": 0.01, "EPISODIC": true, "ABSORBING_GOAL_STATE": true, "TOTAL_TIMESTEPS": }'
