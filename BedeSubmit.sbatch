#!/bin/bash

# set a job name
#SBATCH --job-name=second_wandb_trial_batchv0
# a file for job output, you can check job progress
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:4   # also requests portion of CPU and Memory
#SBATCH --nodes=2
#SBATCH --signal=USR1@60 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=st7ma784@gmail.com
#SBATCH --account=bdlan05
#SBATCH --partition=gpu

source activate open-ce
srun python3 /nobackup/projects/bdlan05/smander3/6DIMCOCO/trainagent.py