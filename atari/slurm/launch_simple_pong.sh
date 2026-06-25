#!/bin/bash

# Usage: bash launch_array.sh <script_name> <suffix> <config_file> <num_seeds> <concurrency_limit>
SCRIPT=$1 # algos/{script}.py, ie just "cov_lstd"
SUFFIX=$2
CONFIG=${3:-"{}"}         # Now $3 matches config_file
SEEDS=${4:-1}             # Now $4 matches num_seeds
CONCURRENCY_LIMIT=${5:-1} # Now $5 matches concurrency_limit

# Define the Date once at launch
DATE_STR=$(date +%Y-%m-%d)

ENVS=(
  "Pong-v5" 
)

export ENVS_LIST="${ENVS[*]}"

# Create the top-level directory before submitting
mkdir -p "slurm/logs/${DATE_STR}"

# Submit and pass DATE_STR as the 5th argument to the slurm script
sbatch --array=0-0%${CONCURRENCY_LIMIT} \
    --cpus-per-task=5 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --time=8:00:00 \
    --output="slurm/logs/${DATE_STR}/%A_%a_job.out" \
    --error="slurm/logs/${DATE_STR}/%A_%a_job.err" \
    slurm/run_slurm_array.sh \
    "$SCRIPT" \
    "$SUFFIX" \
    "$CONFIG" \
    "$SEEDS" \
    "$DATE_STR" 
    
echo "Submitted Array Job. Overrides: $CONFIG"
echo "Date folder: slurm/logs/${DATE_STR}/"

