#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --output=slurm/logs/deepsea_%A_%a.out
#SBATCH --error=slurm/logs/deepsea_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --array=20-21

# ----------------------------
# CONFIG
# ----------------------------

FILE=$1
DATE=$(date +%Y-%m-%d)
N=${SLURM_ARRAY_TASK_ID}
NUM_EPISODES=50000
TOTAL_TIMESTEPS=$((N * NUM_EPISODES))
NUM_STEPS=$((N / 2))

# ----------------------------
# RUN
# ----------------------------

echo "Running $FILE with DEEPSEA_SIZE=$N, TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"

python $FILE \
    --config "{\"ENV_NAME\": \"DeepSea-bsuite\", \"DEEPSEA_SIZE\": $N, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_STEPS\": $NUM_STEPS}" \
    --run_suffix "deepsea_sweep/${DATE}_short/${FILE%.py}_N${N}" \
    --base-config "ds"
