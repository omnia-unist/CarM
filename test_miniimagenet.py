
from torch.utils.data import Dataset
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
import pickle

import random

import os

class MiniImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str='/data', train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = '/data'
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data = []
        self.targets = []

        self.TrainData = []
        self.TrainLabels = []

        train_in = open(root+"/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open(root+"/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open(root+"/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))

        TEST_SPLIT = 1 / 6

        train_data = []
        train_label = []

        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            x_train = cur_x[int(600 * TEST_SPLIT):]
            y_train = cur_y[int(600 * TEST_SPLIT):]
            train_data.append(x_train)
            train_label.append(y_train)

        self.data = np.concatenate(train_data)
        self.targets = np.concatenate(train_label)

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

    def __len__(self):
        return len(self.TrainData)

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        return img,target


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
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("SEED : ", seed)
                
                final_params.seed = seed

        
        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)


        num_task = final_params.num_task_cls_per_task[0]
        num_classes_per_task = final_params.num_task_cls_per_task[1]

        class_order = np.arange(100)

        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)

        print(class_order)

        dataset = MiniImagenet('/data')

        
        continual = Continual(**vars(final_params))

        for task_id in range(num_task):
            label_st = task_id * num_classes_per_task
            for x in range(label_st, label_st + num_classes_per_task):
                print(class_order[x])
                dataset.getTrainData(class_order[x])

                #print(dataset.TrainData)

                for i in range(len(dataset)):
                    img, label = dataset[i]
                    continual.send_stream_data(img, label, task_id)
            continual.train_disjoint(task_id)
