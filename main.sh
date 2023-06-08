#!/bin/bash

#SBATCH --job-name=train
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --partition=rtx6000

conda activate tl
python main.py configs/default.yaml --run_name server
