#!/bin/bash

# Usage: bash launch_array.sh <script_name> <suffix> <config_file> <num_seeds>
SCRIPT=$1
SUFFIX=$2
SEEDS=${3:-5}
CONCURRENCY_LIMIT=${4:-20} 
CONFIG=${5:-"{}"}  # Default to empty JSON string if not provided


# Define the Date once at launch
DATE_STR=$(date +%Y-%m-%d)

ENVS=("Seaquest-v5" "MsPacman-v5")

export ENVS_LIST="${ENVS[*]}"

# Create the top-level directory before submitting
mkdir -p "slurm/logs/${DATE_STR}"

# Submit and pass DATE_STR as the 5th argument to the slurm script
sbatch --array=0-1%${CONCURRENCY_LIMIT} \
    --cpus-per-task=15 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --output="slurm/logs/${DATE_STR}/%A_%a/job.out" \
    --error="slurm/logs/${DATE_STR}/%A_%a/job.err" \
    slurm/run_slurm_array.sh \
    "$SCRIPT" \
    "$SUFFIX" \
    "$CONFIG" \
    "$SEEDS" \
    "$DATE_STR"
    
echo "Submitted Array Job. Overrides: $CONFIG"
echo "Date folder: slurm/logs/${DATE_STR}/"

