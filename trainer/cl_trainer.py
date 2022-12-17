import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from PIL import Image
import numpy as np
import sys
import os
import threading
import time

from agents import *
from utils.data_manager import DataManager
from datasets.test import get_test_set
from datasets.stream import StreamDataset, MultiTaskStreamDataset
from datasets.replay import ReplayDataset
from datasets.dataloader import ContinualDataLoader, ConcatContinualDataLoader
from lib.swap_manager import SwapManager

class Continual(object):
    def __init__(self, gpu_num=0, batch_size=10, epochs=1, rb_size=100, num_workers=0, swap=False,
                opt_name="SGD", lr=0.1, lr_schedule=None, lr_decay=None,
                sampling="reservoir", train_transform=None, test_transform=None, test_set="cifar100", rb_path=None,
                model="resnet18", agent_name="icarl", mode="disjoint", filename=None, samples_per_task = 5000, **kwargs):
        self.data_manager = DataManager()
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.rb_size = rb_size
        self.sampling = sampling
        self.samples_per_task = samples_per_task
        
        self.swap = swap
        self.num_workers = num_workers

        self.test_set = test_set
        if train_transform == None:
            self.set_transform()
        else:
            self.train_transform = train_transform
        print(self.train_transform)

        if test_transform == None:
            self.test_transform = self.train_transform
            if self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                self.test_transform = transforms.Compose([                   
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])

        else:
            self.test_transform = test_transform
        
        self.test_dataset = get_test_set(test_set, data_manager=self.data_manager, test_transform=self.test_transform)

        self.opt_name = opt_name
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay

        self.device = self.get_device(gpu_num)
        self.model = model
        if filename is None:
            self.filename = "{}_{}_{}_{}_batch{}_epoch{}_rb{}_opt{}_lr{}_{}_spt{}_swap{}".format(agent_name, model, test_set, mode,
                                                                                                batch_size, epochs, rb_size, opt_name, 
                                                                                                lr, sampling, samples_per_task, swap)
        else:
            self.filename = filename
        print(self.filename)

        if swap is True and rb_path is None:
            self.rb_path = "data_"+self.filename
            os.makedirs("data_"+self.filename, exist_ok=True)
        else:
            self.rb_path = rb_path
        
        
        self.agent_name = agent_name.lower()
        self.mode = mode

        if self.mode == "non_disjoint":
            self.set_non_disjoint_dataset()
        elif self.mode == "disjoint":
            self.set_disjoint_dataset()

        if self.agent_name is not None:
            self.agent = self.get_agent(self.agent_name, **kwargs)

    
    def get_device(self, gpu_num):
        device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available else 'cpu') 
        return device
    
    def set_transform(self):
        if self.test_set == "cifar100":
            self.train_transform = transforms.Compose([transforms.RandomCrop((32,32),padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        #transforms.ColorJitter(brightness=0.24705882352941178),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

        elif self.test_set == "cifar10":
            self.train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2470, 0.2435, 0.2615))])


        elif self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.ColorJitter(brightness=63/255),
                                            normalize,
                                            ])
        elif self.test_set == "tiny_imagenet":
            self.train_transform = transforms.Compose(
                                                    [transforms.RandomCrop(64, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4802, 0.4480, 0.3975),
                                                                        (0.2770, 0.2691, 0.2821))])
        elif self.test_set == "mini_imagenet":
            self.train_transform = transforms.Compose([transforms.ToTensor()])
        else:
            pass

    def get_agent(self, agent_name, **kwargs):
        dict_disjoint_agents = {
            "icarl" : ICarl,
            "er" : PureER,
            "bic" : BiC,
            #"bic_keepopt" : BiCOpt,
            #"lwf" : LwF,
            "icarl_distill" : ICarl_distill,
            "bic_distill" : BiC_no_distill,
            "derpp_distill" : DarkERPlus_distill,
            "der" : DarkER,
            "derpp" : DarkERPlus,
            "rm" : Rainbow,
            "test" : Test,
            "tiny" : Tiny,
            "gdumb" : GDumb,
            #"aser" : Aser
        }

        dict_non_disjoint_agents = {
            "stream_er" : StreamER
        }
        agent = agent_name.lower()
       
        if self.mode == "non-disjoint":
            if agent not in dict_non_disjoint_agents:
                raise NotImplementedError(
                    "Unknown model {}, must be among {}.".format(agent, list(dict_non_disjoint_agents.keys()))
                )
            return dict_non_disjoint_agents[agent](self.model, self.opt_name, self.lr, self.lr_schedule, self.lr_decay, self.device, self.num_epochs, self.swap,
                                            self.train_transform, self.data_manager, self.stream_dataset, self.replay_dataset, self.cl_dataloader, self.test_set,
                                            self.test_dataset, self.filename, **kwargs)
        elif self.mode == "disjoint":
            if agent not in dict_disjoint_agents:
                raise NotImplementedError(
                    "Unknown model {}, must be among {}.".format(agent, list(dict_disjoint_agents.keys()))
                )
            return dict_disjoint_agents[agent](self.model, self.opt_name, self.lr, self.lr_schedule, self.lr_decay, self.device, self.num_epochs, self.swap,
                                            self.train_transform, self.data_manager, self.stream_dataset, self.replay_dataset, self.cl_dataloader, self.test_set,
                                            self.test_dataset, self.filename, **kwargs)
        

    def set_non_disjoint_dataset(self):
        self.stream_dataset = StreamDataset(batch=self.batch_size, transform=self.train_transform)
        self.replay_dataset = ReplayDataset(rb_path=self.rb_path, rb_size=self.rb_size,
                                            transform=self.train_transform, sampling=self.sampling, agent=self.agent_name)
        self.cl_dataloader = ContinualDataLoader(self.stream_dataset, self.replay_dataset, self.data_manager,
                                                num_workers=self.num_workers, swap=self.swap, batch=self.batch_size)

    def set_disjoint_dataset(self):
        self.train = False
        #self.samples_per_task = 5000
        self.stream_dataset = MultiTaskStreamDataset(batch=self.batch_size,
                                            samples_per_task = self.samples_per_task, 
                                            transform=self.train_transform)
        self.replay_dataset = ReplayDataset(rb_path=self.rb_path, rb_size=self.rb_size,
                                            transform=self.train_transform, sampling=self.sampling, agent=self.agent_name)
        self.cl_dataloader = ConcatContinualDataLoader(self.stream_dataset, self.replay_dataset, self.data_manager,
                                                num_workers=self.num_workers, swap=self.swap, batch=self.batch_size)

        #self.cl_dataloader = MultiTaskContinualDataLoader(self.stream_dataset, self.replay_dataset, self.data_manager,
        #                                        num_workers=self.num_workers, swap=self.swap, batch=self.batch_size)


    def send_stream_data(self, img, label, task_id):
        self.data_manager.append_new_class(label)
        if task_id is not None:
            self.data_manager.append_new_task(task_id, self.data_manager.map_str_label_to_int_label[label])

        if self.mode == "non-disjoint":
            self.stream_dataset.append_stream_data( img, self.data_manager.map_str_label_to_int_label[label], task_id )
            
            if len(self.stream_dataset.data) == self.batch_size:
                print("training start")
                self._worker_event = threading.Event()
                self._worker_thread = threading.Thread(target=self.train_non_disjoint)
                self._worker_thread.daemon = True
                self._worker_thread.start()
                return "Training started"
            else:
                return "Sample added"

        elif self.mode == "disjoint":
            train_task_id, is_train_ok = self.stream_dataset.append_stream_data( img, 
                                                                            self.data_manager.map_str_label_to_int_label[label], 
                                                                            task_id, self.train )        
            if (train_task_id is not None) and (is_train_ok is True):
                self.train_disjoint(train_task_id)
                return "Training started"
            else:
                return "Sample added"


    
    def train_disjoint(self, task_id):
        self.train = True
        
        print("self.data_manager.map_str_label_to_int_label :", self.data_manager.map_str_label_to_int_label)
        print("self.data_manager.map_int_label_to_str_label :", self.data_manager.map_int_label_to_str_label)
        
        self.agent.before_train(task_id) # 여기서 test_dataset, stream_dataset append
        self.agent.train()
        self.agent.after_train() # 여기서 RB update, epoch 모두 끝난상태
        self.train = False

    def train_non_disjoint(self):
        #self.model.train()
        while True:
            for i, (task_id, stream_img, stream_label, replay_img, replay_label) in enumerate(self.cl_dataloader):
                self.agent.train(i, task_id, stream_img, stream_label, replay_img, replay_label)
                

                