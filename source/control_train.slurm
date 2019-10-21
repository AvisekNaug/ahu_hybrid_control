#!/bin/bash

#SBATCH --mail-user=avisek.naug@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3 #no of cpu cores per task
#SBATCH --mem=64G #ram per node
#SBATCH --mem-per-cpu=64G #ram per cpu core #--partition=debug
#SBTACH --output=job_output1.out
#SBATCH --job-name=incremental_rl_Control
#SBATCH --time=4-00:00:00

source /home/nauga/anaconda3/bin/activate SmartBuildings

python hybrid_rl.py
