#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --output=slurm/logs/deepsea/%A_%a.out
#SBATCH --error=slurm/logs/deepsea/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --array=20-50

LOGDIR=slurm/logs/deepsea/${SLURM_ARRAY_JOB_ID}
mkdir -p "$LOGDIR"
# ----------------------------
# CONFIG
# ----------------------------

FILE=$1
DATE=$(date +%Y-%m-%d)
N=${SLURM_ARRAY_TASK_ID}
NUM_EPISODES=50000
TOTAL_TIMESTEPS=$((N * NUM_EPISODES))
EPISODIC=false
N0=1000
# ----------------------------
# RUN
# ----------------------------

echo "Running $FILE with DEEPSEA_SIZE=$N, TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"

python $FILE \
    --config "{\"ENV_NAME\": \"DeepSea-bsuite\", \"DEEPSEA_SIZE\": $N, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"EPISODIC\": $EPISODIC, \"EFFECTIVE_VISITS_TO_REMAIN_OPT\": $N0}" \
    --run_suffix "deepsea_sweep/${DATE}/${FILE%.py}_N${N}_E_${EPISODIC}_N0_${N0}" \
    --base-config "ds"

