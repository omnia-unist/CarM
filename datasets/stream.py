from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
from collections import deque
import time
from queue import Queue
import math

TIMEOUT = 1
SLEEP = 0.001


if __debug__:
    pass
"""
    import logging
    import time
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
"""
class StreamDataset(IterableDataset):
    def __init__(self, batch, transform):
        self.batch = batch
        self.transform = transform
        self.raw_data_dq = deque(list(), self.batch)

        self.data = []
        self.targets = []
        self.tasks = []

        self._recv_data = Queue()

    def append_stream_data(self, img, label, task_id):
        self.put_queue(img,label,task_id)

        self.data.append(img)
        self.targets.append(label)
        self.tasks.append(task_id)

    def put_queue(self, img, label, task_id):
        self._recv_data.put((img,label,task_id))
    
    def __iter__(self):
        while True:
            try:
                img, label, task_id = self._recv_data.get(timeout=TIMEOUT)
            except:
                time.sleep(SLEEP)
                continue

            transformed_img = self.transform(img)

            #print("insert queue!")
            self.raw_data_dq.append(img)

            yield (transformed_img, label, task_id)


class MultiTaskStreamDataset(Dataset):
    def __init__(self, batch, samples_per_task, transform):
        self.batch = batch
        self.samples_per_task = samples_per_task
        self.transform = transform
        self.data_queue = dict()

        self.classes_in_dataset = set()
        self.data = list()
        self.targets = list()

    def append_stream_data(self, img, label, task_id, is_train):
        if task_id not in self.data_queue:
            self.data_queue[task_id] = list()
        self.data_queue[task_id].append((img, label))

        #print(is_train, len(self.data_queue[task_id]))

        """
        if (is_train is False) and len(self.data_queue[task_id]) >= self.samples_per_task:
            print("ready to train..", len(self.data_queue[task_id]))
            return (task_id, True) #self.create_task_dataset(task_id))
        else:
            return (None, False)
        """

        return (None, False)
    
    def clean_stream_dataset(self):
        del self.data[:]
        del self.targets[:]
        
        self.data = list()
        self.targets = list()

    def create_task_dataset(self, task_id):
        if self.data is True and self.targets is True:
            del self.data, self.targets, self.tasks

        self.classes_in_dataset = list()
        self.data = list()
        self.targets = list()

        #for i, (img, label) in enumerate(self.data_queue[task_id]):
        i = 0
        while True:
            if len(self.data_queue[task_id]) == 0: #or len(self.data) >= self.samples_per_task:
                break
            img, label = self.data_queue[task_id][i]
            self.data.append(img)
            self.targets.append(label)
            if label not in self.classes_in_dataset:
                self.classes_in_dataset.append(label)

            del self.data_queue[task_id][i]

        print("stream dataset classes_in dataset : ", self.classes_in_dataset)
        print("stream dataset len : ", len(self.data))

        """
        del self.data_queue[task_id][:self.samples_per_task]
        """
    def split(self, ratio=0.1):
        stream_val_data, stream_val_target, stream_rep_data, stream_rep_target = [],[],[],[]
        stream_val_ac_index, stream_rep_ac_index = [],[]
        
        for new_label in self.classes_in_dataset:
            sub_data, sub_label, actual_index = self.get_sub_data(new_label)
            num_for_val_data = math.ceil(len(sub_data) * ratio)
            print(len(sub_data), ratio, num_for_val_data)

            stream_val_data.extend(sub_data[:num_for_val_data])
            stream_val_target.extend(sub_label[:num_for_val_data])
            stream_val_ac_index.extend(actual_index[:num_for_val_data])

            stream_rep_data.extend(sub_data[num_for_val_data:])
            stream_rep_target.extend(sub_label[num_for_val_data:])
            stream_rep_ac_index.extend(actual_index[num_for_val_data:])
        
        return stream_val_data, stream_val_target, stream_val_ac_index, stream_rep_data, stream_rep_target, stream_rep_ac_index
    
    def get_sub_data(self, label):
        sub_data = []
        sub_label = []
        actual_index = []
        for idx in range(len(self.data)):
            if self.targets[idx] == label:
                sub_data.append(self.data[idx])
                sub_label.append(label)
                actual_index.append(idx)

        return sub_data, sub_label, actual_index
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.transform(img)
        label = self.targets[idx]
        #print("FETCHED  : ", len(self.data), idx)


        """
        if __debug__:
            start = time.perf_counter()
        img = self.data[idx]
        if __debug__:
            get_tar_end = time.perf_counter()
        img = self.transform(img)
        if __debug__:
            trans_end = time.perf_counter()
        label = self.targets[idx]
        if __debug__:
            targets_end = time.perf_counter()
        
        #if __debug__:
        #    log.debug(f"StreamDataset GET_img\t{get_tar_end-start}\t\tAUG_img\t{trans_end-get_tar_end}\t\tGET_tar\t{targets_end-trans_end}\t\tTOTAL_TIME\t{targets_end-start}\t\tINDEX\t{idx}")
        """                   
        return idx, img, label
