#!/bin/bash

# Usage: bash launch_array.sh <script_name> <suffix> <config_file> <num_seeds>
SCRIPT=$1
SUFFIX=$2
CONFIG=$3
SEEDS=${4:-5}
CONCURRENCY_LIMIT=${5:-20} 

# Define the Date once at launch
DATE_STR=$(date +%Y-%m-%d)

ENVS=(
    "Amidar-v5" "Bowling-v5" "BattleZone-v5" "DoubleDunk-v5" 
    "Frostbite-v5" "KungFuMaster-v5" "Riverraid-v5" "NameThisGame-v5"
    "Phoenix-v5" "Qbert-v5" "Asterix-v5" "Breakout-v5" 
    "Freeway-v5" "SpaceInvaders-v5" "Seaquest-v5" "MsPacman-v5"
)

export ENVS_LIST="${ENVS[*]}"

# Create the top-level directory before submitting
mkdir -p "slurm/logs/${DATE_STR}"

# Submit and pass DATE_STR as the 5th argument to the slurm script
sbatch --array=0-15%${CONCURRENCY_LIMIT} \
    --cpus-per-task=15 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --output="slurm/logs/${DATE_STR}/%A_%a/job.out" \
    --error="slurm/logs/${DATE_STR}/%A_%a/job.err" \
    bash_scripts/run_slurm_array.sh \
    "$SCRIPT" \
    "$SUFFIX" \
    "$CONFIG" \
    "$SEEDS" \
    "$DATE_STR"
    
echo "Submitted Array Job for Config: $CONFIG"
echo "Date folder: slurm/logs/${DATE_STR}/"

