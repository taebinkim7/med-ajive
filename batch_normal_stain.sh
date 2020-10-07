#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=16g
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 5-

module add python
python3 batch_normstain.py
