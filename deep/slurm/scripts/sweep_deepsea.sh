#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --output=slurm/logs/deepsea_%A_%a.out
#SBATCH --error=slurm/logs/deepsea_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --array=5-30

# ----------------------------
# CONFIG
# ----------------------------

FILE=$1
DATE=$(date +%Y-%m-%d)
N=${SLURM_ARRAY_TASK_ID}
NUM_EPISODES=10000
TOTAL_TIMESTEPS=$((N * NUM_EPISODES))

# ----------------------------
# RUN
# ----------------------------

echo "Running $FILE with DEEPSEA_SIZE=$N, TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"

OUT=$(python $FILE \
    --config "{\"ENV_NAME\": \"DeepSea-bsuite\", \"DEEPSEA_SIZE\": $N, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS}" \
    --run_suffix "deepsea_sweep/${DATE}/${FILE%.py}_N${N}")

MEAN_RET=$(echo "$OUT" | grep "RESULT" | sed 's/.*mean_return=//')

echo -e "$N\t$MEAN_RET" > ../logs/tmp_result_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv
