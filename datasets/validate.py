from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import torch
from collections import OrderedDict

class ValidateDataset(Dataset):
    def __init__(self, rb_size, transform, sampling):
        #self.rb_path = rb_path
        self.transform = transform
        self.rb_size = rb_size
        self.sampling = sampling

        self.data = list()
        self.targets = list()

        self.offset = dict()
        self.len_per_cls = dict()
    
    def __len__(self):
        #return self.filled_counter
        assert len(self.data) == len(self.targets)

        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.transform(img)
        label = self.targets[idx]

        return idx, img, label