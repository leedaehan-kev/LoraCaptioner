#!/bin/bash

#SBATCH --job-name=attn-only
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda activate tl
python main.py configs/attn-only.yaml --run_name server-64-attn-only
