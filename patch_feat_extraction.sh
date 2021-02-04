#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=9G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:N
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate med-ajive

python scripts/patch_feat_extraction.py
