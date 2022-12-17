#scriptize....

import yaml
import argparse
import os
import torch
import random
from types import SimpleNamespace

from init_tasks import test_cifar10, test_cifar100, test_imagenet100, test_imagenet1000, test_tinyimagenet, test_miniimagenet


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)[key]
        except yaml.YAMLError as exc:
            print(exc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='config/general.yml')
    parser.add_argument('--gpu_num', type=int, default=0)

    args = parser.parse_args()

    params = load_yaml(args.config)
    final_params = SimpleNamespace(**params)
    final_params.gpu_num = args.gpu_num

    print("\n\n==========================================================\n\n")
    print(final_params.result_save_path)
    print("\n")
    print(final_params)
    print("\n\n==========================================================\n\n")


    exp_dataset = {
        'cifar10' : test_cifar10,
        'cifar100' : test_cifar100,
        'imagenet100' : test_imagenet100,
        'imagenet1000' : test_imagenet1000,
        'mini_imagenet' : test_miniimagenet,
        'tiny_imagenet' : test_tinyimagenet
    }

    exp_dataset[final_params.test_set].experiment(final_params)


if __name__ == '__main__':
    main()