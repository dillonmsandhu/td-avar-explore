#!/bin/bash

# 1. Navigate "up" two levels to the 'deep/' directory
# This ensures the job starts in 'deep/', finding the python files and 
# writing logs to 'deep/slurm/logs' correctly.
cd "$(dirname "$0")/../.." || exit

# 2. Run sbatch pointing back to the scripts folder

sbatch slurm/scripts/mc.sh cov_lstd_whiten.py
# sbatch slurm/scripts/mc.sh cov_lstd_next_phi.py

sbatch slurm/scripts/minatar.sh cov_lstd_whiten.py
# sbatch slurm/scripts/minatar.sh cov_lstd_next_phi.py

sbatch slurm/scripts/deepsea.sh cov_lstd_whiten.py
# sbatch slurm/scripts/deepsea.sh cov_lstd_next_phi.py
