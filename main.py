import argparse
import yaml

from src.run import ImageCaptionerRun


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config file')
    parser.add_argument('--project', type=str, default='tl_summer2023', help='project name')
    parser.add_argument('--run_name', type=str, help='run name')
    args = parser.parse_args()

    # read config
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # run
    run = ImageCaptionerRun(config, args.project, args.run_name)
    run()
