from time import perf_counter
from torch.utils.data import Dataset
import torch
import time
import sys

class ConcatDataset(Dataset):
    def __init__(self, stream_dataset, replay_dataset):
        self.replay_dataset = replay_dataset
        self.stream_dataset = stream_dataset
        self.img_sizes = []

    def update_memory_flag(self):
        self.memory_flag = [True]*len(self.replay_dataset) + [False]*len(self.stream_dataset)
        #self.what_to_swap = torch.tensor([False]*(len(self.replay_dataset)+len(self.stream_dataset)))

    def __len__(self):
        return len(self.replay_dataset) + len(self.stream_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.replay_dataset):
            replay_idx, img, label = self.replay_dataset[idx]
        else:
            stream_idx, img, label = self.stream_dataset[idx-len(self.replay_dataset)]
        
        return idx, img, label


# added class 
class GDumbDataset(Dataset):
    def __init__(self, replay_dataset):
        self.replay_dataset = replay_dataset
        
    def update_memory_flag(self):
        self.memory_flag = [True]*len(self.replay_dataset)

    def __len__(self):
        return len(self.replay_dataset)
    
    def __getitem__(self, idx):
        idx, img, label = self.replay_dataset[idx]
        return idx, img, label
