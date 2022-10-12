#!/bin/bash

# set a job name
#SBATCH --job-name=second_wandb_trial_batchv0
# a file for job output, you can check job progress
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:1   # also requests portion of CPU and Memory
#SBATCH --nodes=1
#SBATCH --signal=USR1@60 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=st7ma784@gmail.com
#SBATCH --account=bdlan05
#SBATCH --partition=gpu
export SLURM_NNODES=$SLURM_JOB_NUM_NODES
export CONDADIR=/nobackup/projects/bdlan05/$USER
export wandb='9cf7e97e2460c18a89429deed624ec1cbfb537bc'

source $CONDADIR/miniconda/etc/profile.d/conda.sh
conda activate $CONDADIR/miniconda/envs/open-ce # ...and activate the conda environment

#source activate open-ce
python3 /nobackup/projects/bdlan05/smander3/6DimCOCO/trainagent.py  --data_dir /nobackup/projects/bdlan05/smander3/data