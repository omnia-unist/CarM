from torchvision.datasets import CIFAR10
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


class iCIFAR10(CIFAR10):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super(iCIFAR10,self).__init__(root,
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

    org_filename = None

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
                print("SEED : ", seed)
        
        
        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)

        print(final_params.filename)
        

        num_task = final_params.num_task_cls_per_task[0]
        num_classes_per_task = final_params.num_task_cls_per_task[1]

        class_order = np.arange(10)

        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)
        # order from https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/options/data/imagenet1000_1order.yaml
        elif final_params.data_order == 'fixed':
            if final_params.agent_name == "rm":
                class_order = [9,1,6,0,3,2,5,7,4,8]
            pass
        print(class_order)

        continual = Continual(**vars(final_params))

        dataset = iCIFAR10('./data')
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
        
        del continual
