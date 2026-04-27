#!/bin/bash

# Usage: bash launch_array.sh <script_name> <suffix> <config_file> <num_seeds> <concurrency_limit>
SCRIPT=$1 # algos/{script}.py, ie just "cov_lstd"
SUFFIX=$2
CONFIG=${3:-"{}"}         # Now $3 matches config_file
SEEDS=${4:-4}             # Now $4 matches num_seeds
CONCURRENCY_LIMIT=${5:-20} # Now $5 matches concurrency_limit

# Define the Date once at launch
DATE_STR=$(date +%Y-%m-%d)

ENVS=(
  "Adventure-v5" 
  "Gravitar-v5" 
  "MontezumaRevenge-v5" 
  "Pitfall-v5" 
  "PrivateEye-v5" 
  "Solaris-v5" 
  "Venture-v5"
)

export ENVS_LIST="${ENVS[*]}"

# Create the top-level directory before submitting
mkdir -p "slurm/logs/${DATE_STR}"

# Submit and pass DATE_STR as the 5th argument to the slurm script
sbatch --array=0-6%${CONCURRENCY_LIMIT} \
    --cpus-per-task=15 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
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


# Usage: bash launch_array.sh <script_name> <suffix> <config_file> <num_seeds>
# SCRIPT=$1 # algos/{script}.py, ie just "cov_lstd"
# SUFFIX=$2
# SEEDS=${3:-5}
# CONCURRENCY_LIMIT=${4:-20} 
# CONFIG=${5:-"{}"}  # Default to empty JSON string if not provided


# # Define the Date once at launch
# DATE_STR=$(date +%Y-%m-%d)


# ENVS=(
#   "Adventure-v5" 
#   "Gravitar-v5" 
#   "MontezumaRevenge-v5" 
#   "Pitfall-v5" 
#   "PrivateEye-v5" 
#   "Solaris-v5" 
#   "Venture-v5"
# )


# ENVS=(
#   "Alien-v5" 
#   "Amidar-v5" 
#   "BankHeist-v5" 
#   "Frostbite-v5" 
#   "Hero-v5" 
#   "MsPacman-v5" 
#   "Qbert-v5" 
#   "Surround-v5" 
#   "WizardOfWor-v5" 
#   "Zaxxon-v5" 
#   "Freeway-v5" 
#   "Gravitar-v5" 
#   "MontezumaRevenge-v5" 
#   "Pitfall-v5" 
#   "PrivateEye-v5" 
#   "Solaris-v5" 
#   "Venture-v5"
# )

# export ENVS_LIST="${ENVS[*]}"

# # Create the top-level directory before submitting
# mkdir -p "slurm/logs/${DATE_STR}"

# # Submit and pass DATE_STR as the 5th argument to the slurm script
# sbatch --array=0-6%${CONCURRENCY_LIMIT} \
#     --cpus-per-task=15 \
#     --partition=compsci-gpu \
#     --gres=gpu:a5000:1 \
#     --output="slurm/logs/${DATE_STR}/%A_%a/job.out" \
#     --error="slurm/logs/${DATE_STR}/%A_%a/job.err" \
#     slurm/run_slurm_array.sh \
#     "$SCRIPT" \
#     "$SUFFIX" \
#     "$CONFIG" \
#     "$SEEDS" \
#     "$DATE_STR"
    
# echo "Submitted Array Job. Overrides: $CONFIG"
# echo "Date folder: slurm/logs/${DATE_STR}/"

