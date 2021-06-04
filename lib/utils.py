
import datetime
import logging
import os
import warnings

import torch
from torch.nn import functional as F
from torchvision import transforms


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def extract_features(model, loader, device):

    targets, features = [], []

    state = model.training
    model.eval()


    for inputs, _targets in loader:
        #print(inputs, _targets)
        _targets = _targets.numpy()
        #_features = model.features(inputs.to(device)).detach().cpu().numpy()
        _features = model.feature_extractor(inputs.to(device)).detach().cpu().numpy()
        
        features.append(_features)
        targets.append(_targets)

    model.train(state)

    #print(targets)
    
    return np.concatenate(features), np.concatenate(targets)


def Image_transform(images, transform):
    data = transform(images[0]).unsqueeze(0)
    for index in range(1, len(images)):
        data = torch.cat((data, transform(images[index]).unsqueeze(0)), dim=0)
    return data


def compute_class_mean(images, model, device, transform):
    x = Image_transform(images, transform).to(device)

    with torch.no_grad():
        feature_extractor_output = F.normalize(model.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
    class_mean = np.mean(feature_extractor_output, axis=0)
    return class_mean, feature_extractor_output


def compute_exemplar_class_mean(classes_so_far, data, model, device, base_transform, classify_transform=None):
    class_mean_set = []
    for index in range(classes_so_far):
        print("compute the class mean of %s"%(str(index)))
        exemplar, exp_idx = data.get_sub_data(index)
        #exemplar = data_dict[index]
        print("the number of examplar : ", len(exemplar))
        
        class_mean, _ = compute_class_mean(exemplar, model, device, base_transform)
        class_mean_,_= compute_class_mean(exemplar, model, device, classify_transform)
        class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
        #class_mean = class_mean/np.linalg.norm(class_mean)
        
        class_mean_set.append(class_mean)
    return class_mean_set