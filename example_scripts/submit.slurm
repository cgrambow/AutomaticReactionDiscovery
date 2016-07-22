#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 4
#SBATCH -t 06:00:00
#SBATCH -J test
#SBATCH -o test.o%j

# Load modules if not already available
module load python/2.7-anaconda
module load g09

# Detect hyperthreading
NTHREADS=`scontrol show node | tail | grep "ThreadsPerCore" | awk -F"=" '{split($3,a," ");print a[1]}'`

# Calculate number of cores available
CPUS_PER_NODE=`echo $SLURM_JOB_CPUS_PER_NODE | awk -F"(" '{print $1}'`
((NCORE = SLURM_JOB_NUM_NODES * CPUS_PER_NODE / NTHREADS))

# Create host file
WDIR=`pwd`
(srun bash -c hostname) > hostfile

# Activate Anaconda environment that includes all dependencies
source activate ard
python -m scoop --tunnel --hostfile hostfile -v -n $NCORE ard.py input.txt
source deactivate

# Clean up
cd $WDIR
rm hostfile
