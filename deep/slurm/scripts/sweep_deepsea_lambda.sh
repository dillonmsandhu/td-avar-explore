#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --output=slurm/logs/%A/%a.out
#SBATCH --error=slurm/logs/%A/%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --array=1-10

# ----------------------------
# CONFIG
# ----------------------------
LOGDIR=slurm/logs/${SLURM_ARRAY_JOB_ID}
mkdir -p "$LOGDIR"

FILE=$1
DATE=$(date +%Y-%m-%d)
N=20
NUM_EPISODES=50000
TOTAL_TIMESTEPS=$((N * NUM_EPISODES))
LAMBDA=1/${SLURM_ARRAY_TASK_ID}

# ----------------------------
# RUN
# ----------------------------

LAMBDA=$(awk "BEGIN {print 1/${SLURM_ARRAY_TASK_ID}}")

echo "Running $FILE with DEEPSEA_SIZE=$N and lambda=$LAMBDA, TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"

python $FILE \
  --config "{\"ENV_NAME\": \"DeepSea-bsuite\", \"DEEPSEA_SIZE\": $N, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"GAE_LAMBDA\": $LAMBDA}" \
  --run_suffix "lambda_sweep/L_${LAMBDA}_N_${N}" \
  --base-config "ds"
