#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_fourrooms_dual_lstd.sh
# Optional overrides:
#   TOTAL_TIMESTEPS=500000 N_SEEDS=2 FOURROOMS_SIZE=15 ./run_fourrooms_dual_lstd.sh

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-125000}"
N_SEEDS="${N_SEEDS:-1}"
FOURROOMS_SIZE="${FOURROOMS_SIZE:-13}"
FOURROOMS_GOAL_ROW="${FOURROOMS_GOAL_ROW:-$((FOURROOMS_SIZE-2))}"
FOURROOMS_GOAL_COL="${FOURROOMS_GOAL_COL:-$((FOURROOMS_SIZE-2))}"
FOURROOMS_FAIL_PROB="${FOURROOMS_FAIL_PROB:-0.0}"
SEED="${SEED:-42}"
RND_NETWORK_TYPE="${RND_NETWORK_TYPE:-cnn}"
RUN_SUFFIX="${RUN_SUFFIX:-fourrooms_truev_cnn_$(date +%Y%m%d_%H%M%S)}"

python -m algos.cov_dual_lstd \
  --base-config visual \
  --env_ids FourRoomsCustom-v0 \
  --run_suffix "${RUN_SUFFIX}" \
  --n-seeds "${N_SEEDS}" \
  --save-checkpoint \
  --config "{\"SEED\":${SEED},\"RND_NETWORK_TYPE\":\"${RND_NETWORK_TYPE}\",\"BIAS\":false,\"MIN_LSTD_LR_RI\":0.1,\"FOURROOMS_SIZE\":${FOURROOMS_SIZE},\"FOURROOMS_FAIL_PROB\":${FOURROOMS_FAIL_PROB},\"FOURROOMS_RESAMPLE_INIT_POS\":false,\"FOURROOMS_RESAMPLE_GOAL_POS\":false,\"FOURROOMS_GOAL_POS\":[${FOURROOMS_GOAL_ROW},${FOURROOMS_GOAL_COL}],\"CALC_TRUE_VALUES\":true,\"TOTAL_TIMESTEPS\":${TOTAL_TIMESTEPS}}"
