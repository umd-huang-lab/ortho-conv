#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=orthcnn                                      # sets the job name if not set from environment
#SBATCH --array=1-12                                             # Submit 8 array jobs, throttling to 4 at a time
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

mode=${1}
lr=${2}

expname=mode_${mode}_lr_${lr}

python train.py --mode ${mode} --lr ${lr} > logs/${expname}.log 2>&1

}

source /cmlscratch/huiminz1/anaconda/etc/profile.d/conda.sh
conda activate base

modes=("full" "partial")
lrs=(0.0005 0.001 0.005 0.01 0.02 0.1)

mode_idx=$(( (${SLURM_ARRAY_TASK_ID} - 1) / 6))
lr_idx=$(( (${SLURM_ARRAY_TASK_ID} - 1) % 6))


# runexp       	 mode                lr
runexp        ${modes[$mode_idx]}    ${lrs[$lr_idx]}  
