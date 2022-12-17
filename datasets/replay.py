from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import torch
from collections import OrderedDict
import pickle
from array import array 

import time

class ReplayDataset(Dataset):
    num_file_for_label = dict()
    #num_file_for_label 서버가 종료되도 쓰일수 있게 rb_path에 따로 meta file로 저장해놔야함

    def __init__(self, rb_path, rb_size, transform, sampling, agent):
        self.rb_path = rb_path
        self.transform = transform
        self.rb_size = rb_size
        self.filled_counter = 0
        self.sampling = sampling
        self.agent = agent

        self.data = list()
        self.targets = list()

        self.tracker = list()

        self.offset = dict()
        self.len_per_cls = dict()

        if self.sampling == "reservoir_der":
            self.logits = list()
    
        self.get_img_time = []
        self.get_target_time = []
        self.aug_time = []
        self.total_time = []

    def get_sub_data(self, label):

        if label in self.offset and label in self.len_per_cls:
            st = self.offset[label]
            en = self.offset[label]+self.len_per_cls[label]
            sub_data = self.data[st:en]
            sub_label = self.targets[st:en]
            sub_index = list(range(st,en))
            
            return sub_data, sub_label, sub_index

        else:
            sub_data = []
            sub_index = []
            sub_label = []
            for idx in range(len(self.data)):
                if self.targets[idx] == label:
                    sub_data.append(self.data[idx])
                    sub_label.append(label)
                    sub_index.append(idx)
            return sub_data, sub_label, sub_index

    def __len__(self):
        #print(len(self.data), len(self.targets))
        assert len(self.data) == len(self.targets)

        return len(self.targets)


    def is_filled(self):
        if len(self.data) == 0:
            return False
        else:
            return True

    def __getitem__(self, idx):
        if self.agent in ["der","derpp","derpp_distill", "tiny","aser"]:
            return self.getitem_online(idx)

        else:
            return self.getitem_offline(idx)

    def getitem_offline(self, idx):
        img = self.data[idx]
        img = self.transform(img)
        label = self.targets[idx]

        return idx, img, label
    
    def getitem_online(self, idx):
        #print(self.targets)
        #print("IDX : ", idx)

        img = self.data[idx]

        
        img = self.transform(img)
        label = self.targets[idx]
        data_id = self.tracker[idx]
        
        if self.agent in ["der","derpp_distill","derpp"]:
            logit = self.logits[idx]
            logit = torch.as_tensor(array('f', logit))


            return idx, img, label, logit, data_id

        else:
            return idx, img, label, data_id

    """
    def __getitem__(self, idx):
        #worker_info = torch.utils.data.get_worker_info()
        
        #if worker_info is not None:
        #    print(worker_info.id, idx)

        start = time.perf_counter()
        
        img = self.data[idx]
        
        get_tar_end = time.perf_counter()
        
        img = self.transform(img)
        
        trans_end = time.perf_counter()
        
        label = self.targets[idx]
        
        targets_end = time.perf_counter()

        data_id = self.tracker[idx]
        
       
        self.get_img_time.append(get_tar_end-start)
        self.get_target_time.append(targets_end-trans_end)
        self.aug_time.append(trans_end-get_tar_end)
        self.total_time.append(targets_end-start)


        print(f"get_img time : ", np.mean(np.array(self.get_img_time)))
        
        print(f"get_target time : ", np.mean(np.array(self.get_target_time)))
        
        print(f"aug_time time : ", np.mean(np.array(self.aug_time)))
        
        print(f"total time : ", np.mean(np.array(self.total_time)))
       

        #if __debug__:
        #    log.debug(f"ReplayDataset GET_img\t{get_tar_end-start}\tAUG_img\t{trans_end-get_tar_end}\tGET_tar\t{targets_end-trans_end}\tTOTAL_TIME\t{targets_end-start}\tINDEX\t{idx}")
        

        return idx, img, label, data_id

    
    
    
    def get_ringbuffer_idx(self, idx):
        #print(self.data.keys())
        for label_k in self.data.keys():
            offset = len(self.data[label_k])
            #print(offset, label_k, idx)

            if offset <= idx:
                idx = idx - offset
            else:
                return label_k, idx
                break
    
    
    def __getitem__(self, idx):

        #print(self.sampling)
        if self.sampling == "reservoir":
            img = self.data[idx]
            img = self.transform(img)
            label = self.targets[idx]

        else:
            label, new_idx = self.get_ringbuffer_idx(idx)
            img = self.data[label][new_idx]
            img = self.transform(img)

        return img, label
    """