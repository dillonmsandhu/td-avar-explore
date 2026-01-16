#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --output=slurm/logs/%j/log.out
#SBATCH --error=slurm/logs/%j/log.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

LOGDIR=slurm/logs/${SLURM_ARRAY_JOB_ID}
mkdir -p "$LOGDIR"
# ----------------------------
# CONFIG
# ----------------------------

FILE=$1
DATE=$(date +%Y-%m-%d)
N=50
NUM_EPISODES=50000
TOTAL_TIMESTEPS=$((N * NUM_EPISODES))

# ----------------------------
# RUN
# ----------------------------

echo "Running $FILE with DEEPSEA_SIZE=$N, TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"

python $FILE \
    --config "{\"ENV_NAME\": \"DeepSea-bsuite\", \
    \"DEEPSEA_SIZE\": $N, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \
    \"EPISODIC\": false, \"BONUS_SCALE\": 1.96, \"MIN_COV_LR\": 0.1, \
    \"EFFECTIVE_VISITS_TO_REMAIN_OPT\": 1000, \"A_REGULARIZATION_PER_STEP\": 0.001, \
    \"EPISODIC_GAE\": true, \"EPISODIC_LSTD_A\": true, \" \"}" \
    --base-config "ds" \
    --run_suffix "ep_gae_cont_A"
    