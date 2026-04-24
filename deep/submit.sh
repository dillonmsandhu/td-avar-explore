#!/bin/bash


# Define the list of algorithm scripts
# Add or remove filenames here as needed
ALGOS=(
    "4_16_lstd0.py"
    "4_16_lstd.py"
    "4_16_lstd_opt.py"
    "4_16_lspi0.py"
    "4_16_distill.py"
    "4_16_learned_feats.py"
    "4_16_net.py"
    "4_16_grpo.py"
)

# Loop through the array and submit each job
for ALGO in "${ALGOS[@]}"; do
    echo "Submitting $ALGO with Batch ID: $BATCH_ID"
    sbatch slurm/scripts/run_all.sh "$ALGO" "4_18_all_cont"
    # Optional: small sleep to prevent overwhelming the Slurm scheduler 
    # if the list grows very large
    sleep 0.5
    sbatch slurm/scripts/run_exact.sh "$ALGO" "4_18_exact_cont"
    sleep 0.5
done

echo "All jobs submitted."