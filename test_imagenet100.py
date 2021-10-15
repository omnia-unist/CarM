from torchvision.datasets import ImageNet
import numpy as np
from PIL import Image

from torchvision import transforms
from set_dataset import Continual
import sys
from types import SimpleNamespace
import yaml
import argparse
import os
import torch

import random

def load_data(fpath, num_task, num_classes_per_task):
    data = []
    labels = []

    lines = open(fpath)
    
    for i in range(num_task * num_classes_per_task):
        data.append([])
        labels.append([])

    for line in lines:
        arr = line.strip().split()
        data[int(arr[1])].append(arr[0])
        labels[int(arr[1])].append(int(arr[1]))

    return data, labels


def experiment(final_params):
    runs = final_params.run

    for num_run in range(runs):
        print(f"#RUN{num_run}")
        
        if num_run == 0:
            if hasattr(final_params, 'filename'):
                org_filename = final_params.filename
            else:
                org_filename = ""
        
        final_params.filename = org_filename + f'run{num_run}'

        
        if num_run == 0 and hasattr(final_params, 'rb_path'):
            org_rb_path = final_params.rb_path
            print(final_params.rb_path)
        if hasattr(final_params, 'rb_path'):
            final_params.rb_path = org_rb_path + '/' + f'{final_params.filename}'
            os.makedirs(final_params.rb_path, exist_ok=True)

            print(final_params.rb_path)
        print(final_params.filename)

        
        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)
        
        if hasattr(final_params, 'seed_start'):
            if final_params.seed_start is not None:
                seed = final_params.seed_start + num_run
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("SEED : ", seed)
                
                final_params.seed = seed

        num_task = final_params.num_task_cls_per_task[0]
        num_classes_per_task = final_params.num_task_cls_per_task[1]

        class_order = np.arange(100)

        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)
        # order from https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/options/data/imagenet1000_1order.yaml
        elif final_params.data_order == 'fixed':
            class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]


        print(class_order)

        data_path = 'data/imagenet-100'
        fpath = os.path.join(data_path, 'train.txt')

        data, labels = load_data(fpath, num_task, num_classes_per_task)
        
        continual = Continual(**vars(final_params))

        for task_id in range(num_task):
            label_st = task_id * num_classes_per_task
            for label in range(label_st, label_st + num_classes_per_task):
                print(class_order[label])
                for data_path in data[class_order[label]]:
                    data_path = os.path.join('/data/Imagenet', data_path)
                    with open(data_path,'rb') as f:
                        img = Image.open(f)
                        img = img.convert('RGB')
                    continual.send_stream_data(img, class_order[label], task_id)
                print(f"data {class_order[label]} added")

            print("train!")
            continual.train_disjoint(task_id)
