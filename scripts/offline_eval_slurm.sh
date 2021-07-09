#!/bin/bash

# Parameters
#SBATCH --job-name=sample
#SBATCH --output=/checkpoint/%u/jobs/sample-%j.out
#SBATCH --error=/checkpoint/%u/jobs/sample-%j.err
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --time=15
#SBATCH --cpus-per-task 10

srun --label scripts/offline_eval_single.sh 0 50 nq_dev_0708_ewc_l5_g1_3e-5_e5
