#!/bin/bash

NUM_GPUS=2

#SBATCH --job-name=train
#SBATCH --gpus=$NUM_GPUS
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --partition=rtx6000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tl
python main.py configs/default.yaml --run_name ddp_test --num_gpus $NUM_GPUS
