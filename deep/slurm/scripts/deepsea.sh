#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --output=slurm/logs/%j/log.out
#SBATCH --error=slurm/logs/%j/log.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

LOGDIR=slurm/logs/${SLURM_JOB_ID} 
mkdir -p "$LOGDIR"

# ----------------------------
# CONFIG
# ----------------------------

FILE=$1

DATE=$(date +%Y-%m-%d)
N=50
NUM_EPISODES=20000
TOTAL_TIMESTEPS=$((N * NUM_EPISODES))

echo "Running $FILE with DEEPSEA_SIZE=$N, TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"

python $FILE \
    --config "{\"DEEPSEA_SIZE\": $N, \
            \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS}" \
    --base-config "ds" \
    --run_suffix "${DATE}/LESS_WHITEN"