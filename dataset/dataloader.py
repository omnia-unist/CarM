from torch.utils.data import Dataset, IterableDataset, ConcatDataset, DataLoader, Sampler
from itertools import cycle
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.concat import ConcatDataset, GDumbDataset
import os
from PIL import Image
import numpy as np
import torch
import random
from _utils.sampling import sample_update_to_RB


from lib.swap_manager import SwapManager

class CustomSampler(Sampler[int]):
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        #항상 true여야함 (stream batch와 개수 맞추기 위함)
        self.drop_last = drop_last

    def __iter__(self):
        while(True):
            total_length = len(self.dataset)
            if total_length == 0:
                yield []
        
            else:
                if total_length < self.batch_size:
                    batch_size = total_length
                else:
                    batch_size = self.batch_size

                x = random.sample( range(total_length), batch_size)
                
                #print("SAMPLER : ", len(self.dataset), x)

                #print("CHECK OVERLAP!")
                #SwapManager.is_overlapped(x)
                
                yield x
                #if self.drop_last is False and len(self.dataset) % batch_size != 0:
                #    yield np.random.choice( len(self.dataset), len(self.dataset) % batch_size, replace=False ).tolist()


class ContinualDataLoader(object):
    def __init__(self, stream_dataset, replay_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.stream_dataset = stream_dataset
        self.replay_dataset = replay_dataset
        
        self.data_manager = data_manager
        
        self.stream_dataloader = DataLoader(stream_dataset,
                                        batch_size = self.batch_size,
                                        #num_workers = self.num_workers,
                                        drop_last=True)
            
            
        replay_batch_sampler = CustomSampler(replay_dataset,
                                            batch_size = self.batch_size)

        # incrementally reset replay_buffer
        self.replay_dataloader = DataLoader(replay_dataset,
                                        num_workers = self.num_workers,
                                        batch_sampler = replay_batch_sampler)
    
    def update(self, batched_stream_label):
        for new_data, new_label in zip(self.stream_dataset.raw_data_dq, batched_stream_label):
            if type(new_label) is torch.Tensor:
                new_label = new_label.item()
            sample_update_to_RB(self.replay_dataset, self.data_manager, new_data, new_label)
            self.replay_dataset.store_image(new_data, new_label)
            self.data_manager.increase_observed_samples()
        self.stream_dataset.raw_data_dq.clear()


    def __iter__(self):
        if len(self.replay_dataset) == 0 :
            init_stream_img, init_stream_label, stream_task_id = next(iter(self.stream_dataloader))
            yield stream_task_id, init_stream_img, init_stream_label, None, None
            
            print("UPDATE!!!")
            
            self.update(init_stream_label)


        for stream_data, replay_data in zip(self.stream_dataloader, self.replay_dataloader):
            stream_img, stream_label, stream_task_id = stream_data
            replay_img, replay_label = replay_data
            yield stream_task_id, stream_img, stream_label, replay_img, replay_label
            
            print("UPDATE!!")
            
            self.update(stream_label)


class TinyReplayDataLoader(object):
    def __init__(self, replay_dataset, data_manager, num_workers, batch):
        self.num_workers = num_workers
        self.batch_size = batch
        self.replay_dataset = replay_dataset
        
        replay_batch_sampler = CustomSampler(self.replay_dataset,
                                            batch_size = self.batch_size)
            
        self.replay_dataloader = DataLoader(self.replay_dataset,
                                        batch_sampler = replay_batch_sampler,
                                        num_workers = self.num_workers
                                        )
    def __iter__(self):
        yield from self.replay_dataloader
        #for replay_data in self.replay_dataloader:
        #    yield replay_data

class TinyContinualDataLoader(object):
    def __init__(self, stream_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.stream_dataset = stream_dataset
        
        self.stream_dataloader = DataLoader(self.stream_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle=True,
                                        drop_last=True)
    def __iter__(self):
        print("iteration starts!")
        
        yield from self.stream_dataloader
        #for stream_data in self.stream_dataloader:
        #    stream_idx, stream_img, stream_label = stream_data
        #    yield stream_idx, stream_img, stream_label
            

class DERContinualDataLoader(object):
    def __init__(self, stream_dataset, replay_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.stream_dataset = stream_dataset
        self.replay_dataset = replay_dataset
        self.data_manager = data_manager

    def init_dataloader(self):
        self.stream_dataloader = DataLoader(self.stream_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle=True,
                                        drop_last=True)
    def combined_dataloader(self):
        self.stream_dataloader = DataLoader(self.stream_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle=True,
                                        drop_last=True)

                                        
        replay_batch_sampler = CustomSampler(self.replay_dataset,
                                            batch_size = self.batch_size)
            
        self.replay_dataloader = DataLoader(self.replay_dataset,
                                        batch_sampler = replay_batch_sampler,
                                        num_workers = self.num_workers,
                                        )

    def __iter__(self):
        print("iteration starts!")

        if len(self.replay_dataset) == 0:
            self.init_dataloader()
            init_stream_idx, init_stream_img, init_stream_label = next(iter(self.stream_dataloader))
            yield None, init_stream_idx, init_stream_img, init_stream_label

        self.combined_dataloader()
        print(len(self.replay_dataset))
        for replay_data, stream_data in zip(self.replay_dataloader, self.stream_dataloader):
            #replay_idx, replay_img, replay_label = replay_data
            stream_idx, stream_img, stream_label = stream_data
            
            #yield stream_img, stream_label, replay_img, replay_label
            yield replay_data, stream_idx, stream_img, stream_label
    
class ConcatContinualDataLoader(object):
    def __init__(self, stream_dataset, replay_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.stream_dataset = stream_dataset
        self.replay_dataset = replay_dataset
        self.data_manager = data_manager

        self.concat_dataset = ConcatDataset(self.stream_dataset, self.replay_dataset)
    
    def update_loader(self):
        self.concat_dataset.update_memory_flag()
        print("the size of train set is %s"%(str(len(self.concat_dataset))))


        self.concat_dataloader = DataLoader(self.concat_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        pin_memory = True,
                                        shuffle=True
                                        )
    def __iter__(self):
        #for batch_img, batch_label in concat_dataloader:
        
        yield from self.concat_dataloader
        

# added classes
class GDumbDataLoader(object):
    def __init__(self, stream_dataset, replay_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.stream_dataset = stream_dataset
        self.replay_dataset = replay_dataset
        self.data_manager = data_manager

        self.concat_dataset = GDumbDataset(self.replay_dataset)
    

    def update_loader(self): # gdumb should save in memory before first loading data,
        print("the size of train set is %s"%(str(len(self.concat_dataset))))

        self.concat_dataloader = DataLoader(self.concat_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle=True
                                        )
    def __iter__(self):
        #for batch_img, batch_label in concat_dataloader:
        yield from self.concat_dataloader

class AserDataLoader(object):
    def __init__(self, stream_dataset, replay_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.stream_dataset = stream_dataset
        self.replay_dataset = replay_dataset
        self.data_manager = data_manager

        self.concat_dataset = GDumbDataset(self.stream_dataset)
    
    def update_loader(self): # gdumb should save in memory before first loading data,
        print("the size of train set is %s"%(str(len(self.concat_dataset))))

        self.concat_dataloader = DataLoader(self.concat_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle=True
                                        )

    def __iter__(self):
        yield from self.concat_dataloader



class RainbowReplayDataLoader(object):
    def __init__(self, replay_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.replay_dataset = replay_dataset
        self.data_manager = data_manager
        self.concat_dataset = GDumbDataset(self.replay_dataset)
    

    def update_loader(self): # gdumb should save in memory before first loading data,
        print("the size of train set is %s"%(str(len(self.concat_dataset))))
        self.concat_dataloader = DataLoader(self.concat_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle=True
                                        )
    def __iter__(self):
        #for batch_img, batch_label in concat_dataloader:
        yield from self.concat_dataloader

class RainbowStreamDataLoader(object):
    def __init__(self, stream_dataset, data_manager, num_workers, batch, swap):
        self.num_workers = num_workers
        self.batch_size = batch
        self.swap = swap
        self.stream_dataset = stream_dataset
        self.data_manager = data_manager
        self.concat_dataset = GDumbDataset(self.stream_dataset)
    

    def update_loader(self): # gdumb should save in memory before first loading data,
        print("the size of train set is %s"%(str(len(self.concat_dataset))))
        self.concat_dataloader = DataLoader(self.concat_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle=True
                                        )
    def __iter__(self):
        #for batch_img, batch_label in concat_dataloader:
        yield from self.concat_dataloader
