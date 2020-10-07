#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01-00:00:00


module add pytorch
python scripts/ajive_analysis.py
