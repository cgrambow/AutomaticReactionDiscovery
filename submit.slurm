#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J NAME
#SBATCH -o NAME.o%j

# Load modules if not already available
module load python/2.7-anaconda
module load g09

# This should point to the path where `ard.py` and `tssearch.py` are located
ARD=/global/homes/c/cgrambow/Code/AutomaticReactionDiscovery/

# Activate Anaconda environment that includes all dependencies
source activate ard
python $ARD/tssearch.py input.txt
source deactivate
