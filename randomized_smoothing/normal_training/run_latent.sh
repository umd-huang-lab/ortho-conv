#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=orthcnn_latent                                      # sets the job name if not set from environment
#SBATCH --array=1-6                                             # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output slurm-logs/%x_%A_%a.log                                   # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm-logs/%x_%A_%a.log                                    # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=72:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                                     # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                         # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=TIME_LIMIT,FAIL,ARRAY_TASKS                 # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,

function runexp {

lr=${1}

expname=mode_orthogonal_latent_space_lr_${lr}

python train_latent.py --lr ${lr} > logs/${expname}.log 2>&1

}

source /cmlscratch/huiminz1/anaconda/etc/profile.d/conda.sh
conda activate base

lrs=(0.0005 0.001 0.005 0.01 0.02 0.1)

lr_idx=$(( (${SLURM_ARRAY_TASK_ID} - 1) % 6))

# runexp           lr
runexp      ${lrs[$lr_idx]}  
