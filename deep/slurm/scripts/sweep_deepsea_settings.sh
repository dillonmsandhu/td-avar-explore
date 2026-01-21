#!/bin/bash
#SBATCH --job-name=deepsea_sweep
#SBATCH --output=slurm/logs/%A/%a.out
#SBATCH --error=slurm/logs/%A/%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --array=0-31
# ----------------------------
# CONFIG
# ----------------------------
LOGDIR=slurm/logs/${SLURM_ARRAY_JOB_ID}
mkdir -p "$LOGDIR"
FILE=$1
N=50
NUM_EPISODES=50000
TOTAL_TIMESTEPS=$((N * NUM_EPISODES))

# --- GRID SEARCH ARRAYS ---
# We have 32 combinations (2 * 2 * 2 * 2 * 2)
# Using JSON lowercase for booleans (true/false)

OPT_VISITS_VALS=(100 1000)
EMA_COEFF_VALS=(0.5 1.0)
IGAE_STD_VALS=("false" "true")
AI_WEIGHT_VALS=(1.0 0.1)
EPISODIC_VALS=("true" "false")

# --- INDEX CALCULATION ---
# We use integer division and modulo to map the single ID to 5 dimensions
IDX=$SLURM_ARRAY_TASK_ID

# 1. EFFECTIVE_VISITS (stride 1)
i1=$((IDX % 2))
VISITS=${OPT_VISITS_VALS[$i1]}

# 2. EMA_W_COEFF (stride 2)
i2=$(( (IDX / 2) % 2 ))
EMA=${EMA_COEFF_VALS[$i2]}

# 3. I_GAE_STD (stride 4)
i3=$(( (IDX / 4) % 2 ))
IGAE=${IGAE_STD_VALS[$i3]}

# 4. A_i_weight (stride 8)
i4=$(( (IDX / 8) % 2 ))
AIW=${AI_WEIGHT_VALS[$i4]}

# 5. EPISODIC (stride 16)
i5=$(( (IDX / 16) % 2 ))
EPISODIC=${EPISODIC_VALS[$i5]}

# ----------------------------
# RUN
# ----------------------------
# Construct the config JSON string

CONFIG_JSON="{\"ENV_NAME\": \"DeepSea-bsuite\", \"DEEPSEA_SIZE\": $N, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"EFFECTIVE_VISITS_TO_REMAIN_OPT\": $VISITS, \"EMA_W_COEFF\": $EMA, \"I_GAE_STD\": $IGAE, \"A_i_weight\": $AIW, \"EPISODIC\": $EPISODIC}"

echo "Running $FILE [ID: $IDX] with:"
echo "  VISITS: $VISITS, EMA: $EMA, IGAE: $IGAE, AIW: $AIW, EPISODIC: $EPISODIC"

python $FILE \
--config "$CONFIG_JSON" \
--run_suffix "sweep_settings/E_${EPISODIC}/N0_${VISITS}/ID_${IDX}_N_${N}" \
--base-config "ds"