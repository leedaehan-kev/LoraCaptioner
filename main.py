import argparse
import os

import yaml

import torch.distributed as dist
import torch.multiprocessing as mp

from src.run import ImageCaptionerRun


def main(rank, args, config):
    dist.init_process_group(backend='nccl', rank=rank, world_size=args.num_gpus)
    run = ImageCaptionerRun(config, args.project, args.run_name, rank)
    run()
    dist.destroy_process_group()


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config file')
    parser.add_argument('--project', type=str, default='tl_summer2023', help='project name for wandb')
    parser.add_argument('--run_name', type=str, help='run name for wandb')
    parser.add_argument('--num_gpus', type=int, help='number of gpus to use')
    args = parser.parse_args()

    # read config
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # run
    mp.spawn(main, args=(args, config,), nprocs=args.num_gpus)
