from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
from collections import deque
from lib import utils
from multiprocessing import Manager
from array import array
import random

def sample_update_to_RB(replay_dataset, data_manager, new_data, new_label, *arg):
    sampling = replay_dataset.sampling
    #print(sampling)
    assert sampling in ["reservoir","reservoir_der","ringbuffer", "ringbuffer_fixed"], "Sampling must be either reservoir or ringbuffer"

    if sampling == "reservoir":
        reservoir(replay_dataset, data_manager.num_samples_observed_so_far, new_data, new_label)
    
    elif sampling == "reservoir_der":
        reservoir_der(replay_dataset, data_manager.num_samples_observed_so_far, new_data, new_label, *arg)

    elif sampling == "ringbuffer_fixed":
        ringbuffer_fixed(replay_dataset, data_manager.num_samples_observed_so_far, new_data, new_label, *arg)

    elif sampling == "ringbuffer":
        ringbuffer(replay_dataset, data_manager.num_samples_observed_so_far, new_data, new_label)


def multi_task_sample_update_to_RB(replay_dataset, stream_dataset, *arg):
    sampling = replay_dataset.sampling
    print(sampling)
    assert sampling in ["herding", "ringbuffer", "greedy_balance"], "Sampling must be either reservoir, ringbuffer, or greedy_balance" # changed

    if sampling == "herding":
        herding(replay_dataset, stream_dataset, *arg)
    
    elif sampling == "ringbuffer":
        return ringbuffer_offline(replay_dataset, stream_dataset, *arg)

    elif sampling == "greedy_balance": # added
        for new_data, new_label in zip(stream_dataset.data, stream_dataset.targets): # added
            greedy_balance(replay_dataset, new_data, new_label) # added
    
    #elif sampling == "ringbuffer":
    #    for new_data, new_label in zip(stream_dataset.data, stream_dataset.targets):
    #        ringbuffer(replay_dataset, new_data, new_label)
    

def reservoir_der(dataset, obs_id, new_data, new_label, new_logit):
    if dataset.rb_size > obs_id:
        #print("APPENDED")

        dataset.data.append(new_data)
        dataset.targets.append(new_label)
        new_logit = bytes(array(('f') ,new_logit.tolist()))
        dataset.logits.append(new_logit)
        
        dataset.tracker.append(obs_id)

    else:
        idx_random = np.random.randint(0, obs_id)

        if idx_random < dataset.rb_size:
            #print(f"{idx_random} REPLACED")
            
            dataset.data[ idx_random ] = new_data
            dataset.targets[ idx_random ] = new_label
            new_logit = bytes(array(('f') ,new_logit.tolist()))
            dataset.logits[ idx_random ] = new_logit

            dataset.tracker[ idx_random ] = obs_id


def reservoir(dataset, obs_id, new_data, new_label):
    if dataset.rb_size > obs_id:
        dataset.data.append(new_data)
        dataset.targets.append(new_label)
        dataset.tracker.append(obs_id)

    else:
        idx_random = np.random.randint(0, obs_id)
        if idx_random < dataset.rb_size:
            dataset.data[ idx_random ] = new_data
            dataset.targets[ idx_random ] = new_label
            dataset.tracker[ idx_random ] = obs_id


def ringbuffer_fixed(dataset, obs_id, new_data, new_label, test_set):
    if test_set == "cifar10":
        total_classes = 10
    elif test_set in ["cifar100", "imagenet100"]:
        total_classes = 100
    elif test_set in ["imagenet", "imagenet1000"]:
        total_classes = 1000

    memory_per_class = int(dataset.rb_size / total_classes)

    #eviction
    #lock?
    for k in dataset.offset:
        if dataset.len_per_cls[k] > memory_per_class:
            num_samples_to_evict = dataset.len_per_cls[k] - memory_per_class
            del dataset.data [dataset.offset[k] : dataset.offset[k] + num_samples_to_evict]
            del dataset.targets [dataset.offset[k] : dataset.offset[k] + num_samples_to_evict]
            del dataset.tracker [dataset.offset[k] : dataset.offset[k] + num_samples_to_evict]
            
            dataset.len_per_cls[k] = memory_per_class

            for k_2 in dataset.offset:
                if dataset.offset[k_2] > dataset.offset[k]: #dataset.offset[new_label]:
                    dataset.offset[k_2] = dataset.offset[k_2] - num_samples_to_evict

    #insertion
    if new_label not in dataset.offset or new_label not in dataset.len_per_cls:
        dataset.offset.update({new_label : len(dataset.data)})
        #dataset.offset[new_label] = len(dataset.data)
        dataset.len_per_cls[new_label] = 0

    #lock?
    if dataset.len_per_cls[ new_label ] < memory_per_class:
        dataset.data.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label] , new_data)
        dataset.targets.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label] , new_label)
        dataset.tracker.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label] , obs_id)
        
        
        dataset.len_per_cls[new_label] += 1
        
        for k in dataset.offset:
            if dataset.offset[k] > dataset.offset[new_label]:
                dataset.offset[k] += 1

    elif memory_per_class > 0 and dataset.len_per_cls[ new_label ] == memory_per_class:
        
        dataset.data[ dataset.offset[new_label] : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] = \
            dataset.data[ dataset.offset[new_label]+1 : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] + [new_data] 
        
        dataset.targets[ dataset.offset[new_label] : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] = \
            dataset.targets[ dataset.offset[new_label]+1 : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] + [new_label] 
        
        dataset.tracker[ dataset.offset[new_label] : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] = \
            dataset.tracker[ dataset.offset[new_label]+1 : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] + [obs_id] 
        
        
    else:
        assert dataset.len_per_cls[ new_label ] <= memory_per_class, "length per class in RB exceeded"

    #print("curr_label : ", dataset.targets)
    #print("curr_id : ", dataset.tracker)
    #print("len of dataset.data : ", len(dataset.data))
    #print(dataset.targets)
    #print(dataset.offset)
    #print(dataset.len_per_cls)


def ringbuffer(dataset, obs_id, new_data, new_label):
    if new_label not in dataset.offset or new_label not in dataset.len_per_cls:
        #new_label is added
        num_trained_class = len(dataset.offset) + 1
    else:
        num_trained_class = len(dataset.offset)
    
    memory_per_class = int(dataset.rb_size / num_trained_class)

    #eviction
    #lock?
    for k in dataset.offset:
        if dataset.len_per_cls[k] > memory_per_class:
            num_samples_to_evict = dataset.len_per_cls[k] - memory_per_class
            del dataset.data [dataset.offset[k] : dataset.offset[k] + num_samples_to_evict]
            del dataset.targets [dataset.offset[k] : dataset.offset[k] + num_samples_to_evict]
            del dataset.tracker [dataset.offset[k] : dataset.offset[k] + num_samples_to_evict]
            
            dataset.len_per_cls[k] = memory_per_class

            for k_2 in dataset.offset:
                if dataset.offset[k_2] > dataset.offset[k]: #dataset.offset[new_label]:
                    dataset.offset[k_2] = dataset.offset[k_2] - num_samples_to_evict

    #insertion
    if new_label not in dataset.offset or new_label not in dataset.len_per_cls:
        dataset.offset[new_label] = len(dataset.data)
        dataset.len_per_cls[new_label] = 0

    #lock?
    if dataset.len_per_cls[ new_label ] < memory_per_class:
        dataset.data.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label] , new_data)
        dataset.targets.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label] , new_label)
        dataset.tracker.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label] , obs_id)
        
        
        dataset.len_per_cls[new_label] += 1
        
        for k in dataset.offset:
            if dataset.offset[k] > dataset.offset[new_label]:
                dataset.offset[k] += 1

    elif memory_per_class > 0 and dataset.len_per_cls[ new_label ] == memory_per_class:
        
        dataset.data[ dataset.offset[new_label] : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] = \
            dataset.data[ dataset.offset[new_label]+1 : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] + [new_data] 
        
        dataset.targets[ dataset.offset[new_label] : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] = \
            dataset.targets[ dataset.offset[new_label]+1 : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] + [new_label] 
        
        dataset.tracker[ dataset.offset[new_label] : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] = \
            dataset.tracker[ dataset.offset[new_label]+1 : dataset.offset[new_label]+dataset.len_per_cls[new_label] ] + [obs_id] 
        
        
    else:
        assert dataset.len_per_cls[ new_label ] <= memory_per_class, "length per class in RB exceeded"


    #print("curr_label : ", new_label)
    #print("len of dataset.data : ", len(dataset.data))
    #print(dataset.targets)
    #print(dataset.offset)
    #sprint(dataset.len_per_cls)

def ringbuffer_offline(replay_dataset, stream_dataset, val=False):

    if val == True:
        eviction_idx_list = []
    
    num_new_label = 0
    for label in stream_dataset.classes_in_dataset:
        if label not in replay_dataset.offset or label not in replay_dataset.len_per_cls:
            num_new_label += 1
    
    memory_per_class = int(replay_dataset.rb_size / (len(replay_dataset.offset) + num_new_label))
    print("MEM PER CLS : ", memory_per_class)
    #deletion
    for k in replay_dataset.offset:
        if replay_dataset.len_per_cls[k] > memory_per_class:
            num_samples_to_evict = replay_dataset.len_per_cls[k] - memory_per_class
            del replay_dataset.data [replay_dataset.offset[k] : replay_dataset.offset[k] + num_samples_to_evict]
            del replay_dataset.targets [replay_dataset.offset[k] : replay_dataset.offset[k] + num_samples_to_evict]
            #del replay_dataset.tracker [replay_dataset.offset[k] : replay_dataset.offset[k] + num_samples_to_evict]
            
            replay_dataset.len_per_cls[k] = memory_per_class

            for k_2 in replay_dataset.offset:
                if replay_dataset.offset[k_2] > replay_dataset.offset[k]:
                    replay_dataset.offset[k_2] = replay_dataset.offset[k_2] - num_samples_to_evict
    
    #insertion
    for new_label in stream_dataset.classes_in_dataset:
        sub_stream_data, sub_stream_label, sub_stream_idx = stream_dataset.get_sub_data(new_label)

        if new_label not in replay_dataset.offset or new_label not in replay_dataset.len_per_cls:
            replay_dataset.offset[new_label] = len(replay_dataset.data)
            replay_dataset.data.extend( sub_stream_data[-memory_per_class:] )
            replay_dataset.targets.extend( sub_stream_label[-memory_per_class:] )
            replay_dataset.len_per_cls[new_label] = len(sub_stream_label[-memory_per_class:])
            
            if val==True:
                eviction_idx_list.extend(sub_stream_idx[-memory_per_class:])
        else:
            #rewrite
            if replay_dataset.len_per_cls[new_label] == memory_per_class:                
                if len(sub_stream_data) < memory_per_class:
                    replay_dataset.data[ replay_dataset.offset[new_label] : replay_dataset.offset[new_label] + replay_dataset.len_per_cls[new_label] ] = \
                        replay_dataset.data[ -( memory_per_class - len(sub_stream_data) ) : ] + sub_stream_data
                    
                    if val==True:
                        eviction_idx_list.extend(sub_stream_idx)

                else:
                    replay_dataset.data[ replay_dataset.offset[new_label] : replay_dataset.offset[new_label] + replay_dataset.len_per_cls[new_label] ] = \
                        sub_stream_data[-memory_per_class:]
                    
                    if val==True:
                        eviction_idx_list.extend(sub_stream_idx[-memory_per_class:])

            #insert
            elif replay_dataset.len_per_cls[new_label] < memory_per_class:
                print("SMALLER LEN")

                for i in range(len(sub_stream_data)):
                    new_data = sub_stream_data[len(sub_stream_data)-i-1]
                    replay_dataset.data.insert(replay_dataset.offset[new_label] + replay_dataset.len_per_cls[new_label] , new_data)
                    replay_dataset.targets.insert(replay_dataset.offset[new_label] + replay_dataset.len_per_cls[new_label] , new_label)
                    if val==True:
                        eviction_idx_list.extend(sub_stream_idx[len(sub_stream_data)-i-1])

                    replay_dataset.len_per_cls[new_label] += 1
                
                    for k in replay_dataset.offset:
                        if replay_dataset.offset[k] > replay_dataset.offset[new_label]:
                            replay_dataset.offset[k] += 1

                    if replay_dataset.len_per_cls[new_label] >= memory_per_class:
                        break    

            else:
                assert replay_dataset.len_per_cls[ new_label ] <= memory_per_class, "length per class in RB exceeded"


    
    print("VAL IS TRUE ? ", val)
    print("len of dataset.data and target : ", len(replay_dataset.data), len(replay_dataset.targets))
    #print("replay TARGET : ", replay_dataset.targets)
    #print("replay OFFSET : ", replay_dataset.offset)
    #print("replay LEN_PER_CLS : ", replay_dataset.len_per_cls)

    if val==True:
        return eviction_idx_list
    


def herding(replay_dataset, stream_dataset, model, device, classes_so_far, transform):
    
    memory_per_class = int(replay_dataset.rb_size / classes_so_far)

    #eviction

    """
    for i in range(len(replay_dataset.offset)-1,-1,-1):
        if replay_dataset.len_per_cls[i] > memory_per_class:
            num_samples_to_evict = replay_dataset.len_per_cls[i] - memory_per_class
            del replay_dataset.data [replay_dataset.offset[i] : replay_dataset.offset[i] + num_samples_to_evict]
            del replay_dataset.targets [replay_dataset.offset[i] : replay_dataset.offset[i] + num_samples_to_evict]
            replay_dataset.len_per_cls[i] = memory_per_class
        
            for j in range(i+1, len(replay_dataset.offset)):
                replay_dataset.offset[j] = replay_dataset.offset[j] - num_samples_to_evict
    """
    for k in replay_dataset.offset:
        if replay_dataset.len_per_cls[k] > memory_per_class:
            num_samples_to_evict = replay_dataset.len_per_cls[k] - memory_per_class
            evict_end = replay_dataset.offset[k] + replay_dataset.len_per_cls[k]
            evict_start = evict_end - num_samples_to_evict
            del replay_dataset.data [evict_start : evict_end]
            del replay_dataset.targets [evict_start : evict_end]
            #del replay_dataset.tracker [replay_dataset.offset[k] : replay_dataset.offset[k] + num_samples_to_evict]
            
            replay_dataset.len_per_cls[k] = memory_per_class

            for k_2 in replay_dataset.offset:
                if replay_dataset.offset[k_2] > replay_dataset.offset[k]:
                    replay_dataset.offset[k_2] = replay_dataset.offset[k_2] - num_samples_to_evict
                
    for new_label in stream_dataset.classes_in_dataset:
        if new_label not in replay_dataset.offset or new_label not in replay_dataset.len_per_cls:
            if new_label > 0:
                replay_dataset.offset[new_label] = replay_dataset.offset[new_label-1] + replay_dataset.len_per_cls[new_label-1]
            else:
                replay_dataset.offset[new_label] = 0
            replay_dataset.len_per_cls[new_label] = 0

        images, labels, idxs = stream_dataset.get_sub_data(new_label)
        
        #consturct_exemplar_set
        class_mean, feature_extractor_output = utils.compute_class_mean(images, model, device, transform)

        now_class_mean = np.zeros((1,feature_extractor_output.shape[1]))
        #now_class_mean = np.zeros((classes_so_far, model.features_dim))

        for i in range(memory_per_class):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            #exemplar.append(images[index])
            #print("offset : ", replay_dataset.offset, replay_dataset.len_per_cls)

            replay_dataset.data.insert(replay_dataset.offset[new_label] + replay_dataset.len_per_cls[new_label], images[index])
            replay_dataset.targets.insert(replay_dataset.offset[new_label] + replay_dataset.len_per_cls[new_label], new_label)
            replay_dataset.len_per_cls[new_label] += 1
            for i in range(new_label+1, len(replay_dataset.offset)):
                replay_dataset.offset[i] += 1

        #print(new_class)
        #print("the size of exemplar :%s" % (str(len(replay_dataset.get_sub_data(new_label)[0]))))


# added function
# greedy balancing the memory samples for gdumb agent
def greedy_balance(dataset, new_data, new_label):
    # if the label is of a new class make a offset entry and dataset.len_per_cls entry for it.
    if new_label not in dataset.offset:
        dataset.len_per_cls[new_label] = 0
        if new_label > 0:
            dataset.offset[new_label] = dataset.offset[new_label - 1] + dataset.len_per_cls[new_label - 1]
        else:
            dataset.offset[new_label] = 0
    # find the maximum number of samples which can be present in a class
    memory_per_class = dataset.rb_size // max(1, len(dataset.offset))
    # check if the new label needs to be saved
    if(dataset.len_per_cls[new_label] < memory_per_class):
        # check if the memory is full, remove elements if so
        while(len(dataset.data) >= dataset.rb_size):
            index = 0
            # find the class with max number of elements
            index = max(dataset.len_per_cls, key=dataset.len_per_cls.get)
            # select a random sample from the max elements slot and pop it
            idx = random.randrange(dataset.offset[index], (dataset.offset[index] + dataset.len_per_cls[index]))
            dataset.data.pop(idx)
            dataset.targets.pop(idx)
            dataset.len_per_cls[index] -= 1
            offset_adjust(dataset, adjust_insert = 0, label = index)
        # add the new_data to the memory
        dataset.data.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label], new_data)
        dataset.targets.insert(dataset.offset[new_label] + dataset.len_per_cls[new_label], new_label)
        dataset.len_per_cls[new_label] += 1
        offset_adjust(dataset, adjust_insert = 1, label = new_label)
    return

def offset_adjust(dataset, adjust_insert = 1, label = None):
    # change the offset of other classes when samples are added or deleted
    if(adjust_insert):
        # increase offsets of preceding classses
        for i in range(label + 1, len(dataset.offset)):
            dataset.offset[i] += 1
    else:
        # decrease offsets of preceding classses
        for i in range(label + 1, len(dataset.offset)):
            dataset.offset[i] -= 1    
    return