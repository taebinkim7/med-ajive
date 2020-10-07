#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 04-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

module add pytorch
python scripts/patch_feat_extraction.py
