from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch

class DataManager(object):
    #map_str_label_to_int_label = dict()
    #map_int_label_to_str_label = dict()
    #classes_per_task = dict()
    #map_label_to_one_hot 서버가 종료되도 쓰일수 있게 rb_path에 저장해놔야함
    #num_classes_streamed_so_far = 0
    #num_task_so_far = 0

    
    def __init__(self):
        self.num_samples_observed_so_far = 0
        self.map_str_label_to_int_label = dict()
        self.map_int_label_to_str_label = dict()
        self.classes_per_task = dict()
        #map_label_to_one_hot 서버가 종료되도 쓰일수 있게 rb_path에 저장해놔야함
        self.num_classes_streamed_so_far = 0
        self.num_task_so_far = 0

        

    def increase_observed_samples(self):
        self.num_samples_observed_so_far += 1
    
    
    def append_new_class(self, new_class):
        if new_class not in self.map_str_label_to_int_label:
            self.map_str_label_to_int_label[ new_class ] = self.num_classes_streamed_so_far
            self.map_int_label_to_str_label[ self.num_classes_streamed_so_far ] = new_class
            self.num_classes_streamed_so_far += 1
    
    def append_new_task(self, task_id, int_label):
        if task_id not in self.classes_per_task:
            self.classes_per_task[task_id] = list()

        if int_label not in self.classes_per_task[task_id]:
            self.classes_per_task[task_id].append(int_label)

        #print(label, cls.classes_per_task[task_id])
        self.num_task_so_far += 1