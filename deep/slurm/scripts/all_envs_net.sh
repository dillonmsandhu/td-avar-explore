#!/bin/bash

# 1. Navigate "up" two levels to the 'deep/' directory
# This ensures the job starts in 'deep/', finding the python files and 
# writing logs to 'deep/slurm/logs' correctly.
cd "$(dirname "$0")/../.." || exit

# 2. Run sbatch pointing back to the scripts folder
sbatch slurm/scripts/mc.sh cov_net.py 'phi' true
sbatch slurm/scripts/mc.sh cov_net.py 'next_phi' true
sbatch slurm/scripts/mc.sh cov_net.py 'phi' false
sbatch slurm/scripts/mc.sh cov_net.py 'next_phi' false

sbatch slurm/scripts/minatar.sh cov_net.py 'phi' true
sbatch slurm/scripts/minatar.sh cov_net.py 'next_phi' true
sbatch slurm/scripts/minatar.sh cov_net.py 'phi' false
sbatch slurm/scripts/minatar.sh cov_net.py 'next_phi' false

sbatch slurm/scripts/deepsea.sh cov_net.py 'phi' true
sbatch slurm/scripts/deepsea.sh cov_net.py 'next_phi' true
sbatch slurm/scripts/deepsea.sh cov_net.py 'phi' false
sbatch slurm/scripts/deepsea.sh cov_net.py 'next_phi' false
