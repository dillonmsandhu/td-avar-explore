#!/bin/bash
#SBATCH --job-name=atari-array
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=96:00:00

# 1. Arguments
SCRIPT=$1
SUFFIX=$2
CONFIG=$3
NUM_SEEDS=$4
DATE_STR=$5

# 2. Corrected JAX Memory & Deterministic Settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false  
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export TF_XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
export TF_CUDNN_DETERMINISTIC="1"

# 3. Log Directory Setup
LOG_BASE="slurm/logs/${DATE_STR}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$LOG_BASE"

# 4. Environment Mapping
IFS=' ' read -r -a ENVS_ARRAY <<< "$ENVS_LIST"
ENV_NAME=${ENVS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# 5. Metadata for Identification
cat <<EOF > "${LOG_BASE}/metadata.txt"
Job ID: ${SLURM_ARRAY_JOB_ID}
Array Task: ${SLURM_ARRAY_TASK_ID}
Env: ${ENV_NAME}
Config: ${CONFIG}
Suffix: ${SUFFIX}
Date: $(date)
EOF

# --- MONITORING START ---
MONITOR_FLAG="${LOG_BASE}/keep_monitoring"
touch "$MONITOR_FLAG"

# GPU Monitor Loop
(
  echo "timestamp, mem_used, mem_total, mem_util, gpu_util"
  while [ -f "$MONITOR_FLAG" ]; do
    timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -v ts="$timestamp" -F',' '{
      mem_util=int(($1/$2)*100);
      print ts ", " $1 ", " $2 ", " mem_util ", " $3
    }'
    sleep 60
  done
) > "${LOG_BASE}/gpu_usage.log" &

# CPU Monitor Loop
(
  echo "timestamp, cpu_util"
  while [ -f "$MONITOR_FLAG" ]; do
    timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    util=$(top -bn2 -d 0.5 | grep "Cpu(s)" | tail -n 1 | awk '{print 100 - $8}')
    echo "$timestamp, $util"
    sleep 60
  done
) > "${LOG_BASE}/cpu_usage.log" &
# --- MONITORING END ---

# 6. Launch Training
THREADS_PER_SEED=$((SLURM_CPUS_PER_TASK / NUM_SEEDS))

# FIX 2: Lock down greedy CPU libraries per process
export OMP_NUM_THREADS=$THREADS_PER_SEED
export OPENBLAS_NUM_THREADS=$THREADS_PER_SEED
export MKL_NUM_THREADS=$THREADS_PER_SEED
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_SEED
export NUMEXPR_NUM_THREADS=$THREADS_PER_SEED

SCRIPT_NAME=$(basename $SCRIPT)

TRAINING_PIDS=()

for ((i=0; i<NUM_SEEDS; i++)); do
    SEED=$i 
    # Use the safe SCRIPT_NAME for the log file
    SEED_LOG="${LOG_BASE}/${SCRIPT_NAME}_${ENV_NAME}_s${SEED}.log"
    
    # Use the full SCRIPT path for Python
    python -m algos.${SCRIPT} \
        --config ${CONFIG} \
        --run-suffix ${SUFFIX}_s${SEED} \
        --seed ${SEED} \
        --threads ${THREADS_PER_SEED} \
        --envs ${ENV_NAME} > "$SEED_LOG" 2>&1 &
    
    TRAINING_PIDS+=($!)
    sleep 30 # Staggered start is CRITICAL for memory allocation
done 

# Wait for training to finish
wait ${TRAINING_PIDS[*]}

# Cleanup monitoring
rm -f "$MONITOR_FLAG"
sleep 5