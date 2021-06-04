from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
from torchvision import transforms
from set_dataset import Continual
import sys

from types import SimpleNamespace
import yaml
import argparse
import torch
import random

import os

class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTrainData(self,label):
        datas,labels=[],[]

        data=self.data[np.array(self.targets)==label]
        datas.append(data)
        labels.append(np.full((data.shape[0]),label))
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)
        
    def __getitem__(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        return img,target

    def __len__(self):
        return len(self.TrainData)

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


        
        if hasattr(final_params, 'seed_start'):
            if final_params.seed_start is not None:
                seed = final_params.seed_start + num_run
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("SEED : ", seed)


        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)

        print(final_params.filename)

        num_task = final_params.num_task_cls_per_task[0]
        num_classes_per_task = final_params.num_task_cls_per_task[1]

        class_order = np.arange(100)

        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)
        # order from https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/options/data/imagenet1000_1order.yaml
        elif final_params.data_order == 'fixed':
            #class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]

            
            class_order = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
                            94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
                            84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
                            69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
                            17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
                            1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
                            38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
                            60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
                            40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
                            98, 13, 99,  7, 34, 55, 54, 26, 35, 39]
            
        print(class_order)

        continual = Continual(**vars(final_params))

        dataset = iCIFAR100('./data')
        for task_id in range(num_task):
            label_st = task_id * num_classes_per_task
            for x in range(label_st, label_st + num_classes_per_task):
                print(class_order[x])
                dataset.getTrainData(class_order[x])

                #print(dataset.TrainLabels)

                for i in range(len(dataset)):
                    img, label = dataset[i]
                    continual.send_stream_data(img, label, task_id)
            continual.train_disjoint(task_id)
