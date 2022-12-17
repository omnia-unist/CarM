#

from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim

import numpy as np
import copy

import os
from agents.base import Base
from utils.sampling import multi_task_sample_update_to_RB
from lib import utils
import gc
import math
import time

from datasets.validate import ValidateDataset

class SplitedStreamDataset(Dataset):
    def __init__(self, data, targets, ac_idx=None):
        self.data = data
        self.targets = targets
        self.actual_idx = ac_idx
        self.classes_in_dataset = set(targets)
        print("CLASSES IN DATASET : ", self.classes_in_dataset)
    def __len__(self):
        assert len(self.data) == len(self.targets)
        return self.data
    def get_sub_data(self, label):
        sub_data = []
        sub_label = []
        sub_idx = []
        for idx in range(len(self.data)):
            if self.targets[idx] == label:
                sub_data.append(self.data[idx])
                sub_label.append(label)
                sub_idx.append(self.actual_idx[idx])
        return sub_data, sub_label, sub_idx

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())


class BiC(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)
        
        #self.scaler = torch.cuda.amp.GradScaler() 
            
        if test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            val_transform = transforms.Compose([
                                                    #transforms.Resize(256),
                                                    #transforms.CenterCrop(224),                                               
                                                                                                
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.ColorJitter(brightness=63/255),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    self.nomalize,
                                                ])

        else:
            val_transform = transforms.Compose([
                                            transforms.RandomCrop((32,32),padding=4),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            self.nomalize])

        self.classes_so_far = 0
        self.tasks_so_far = 0

        self.soft_incremental_top1_acc = list()
        self.soft_incremental_top5_acc = list()
        
        self.num_swap = list()
        self.bias_layers = list()

        self.criterion = nn.CrossEntropyLoss()
        #self.swap_manager.swap_loss = nn.CrossEntropyLoss(reduction="none")

        self.old_model = None
        #
        # add validation set
        #
        self.val_dataset = ValidateDataset(int(self.replay_dataset.rb_size * 0.1), val_transform, sampling="ringbuffer")
        
        self.replay_dataset.rb_size = int(self.replay_dataset.rb_size * 0.9)

        if 'distill' in kwargs:
            self.distill = kwargs['distill']
        else:
            self.distill = True
        print("======================== DISTILL : ",self.distill)

        
        self.how_long_stay = list()
        self.how_much_reused = list()

        
        if 'dynamic' in kwargs:
            self.dynamic = kwargs['dynamic']
        else:
            self.dynamic = False

    def before_train(self, task_id):
        
        self.curr_task_iter_time = []

        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)

        self.alpha = self.classes_so_far / (self.classes_so_far + len(self.stream_dataset.classes_in_dataset))
        
        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        print("classes_so_far : ", self.classes_so_far)

        self.tasks_so_far += 1
        print("tasks_so_far : ", self.tasks_so_far)
        
        
        """
        if self.tasks_so_far > 1 and (self.DP == True) and self.test_set in ["imagenet", "imagenet1000"]:
            self.model.module.Incremental_learning(self.classes_so_far)

        else:
            self.model.Incremental_learning(self.classes_so_far)        
        """
        
        self.model.Incremental_learning(self.classes_so_far)
        self.model.train()
        self.model.to(self.device)

        self.replay_size = len(self.replay_dataset)
        print("length of replay dataset : ", self.replay_size)
        
        self.stayed_iter = [0] * self.replay_size
        self.num_called = [0] * self.replay_size

        self.stream_losses, self.replay_losses = list(), list()

        #
        # add bias correction layer
        #
        bias_layer = BiasLayer().to(self.device)
        self.bias_layers.append(bias_layer)

        #
        # update validation set (val set should contain data of new classes before training)
        #

        
        if self.swap is True:
            self.swap_manager.before_train()
             
            for new_label in self.stream_dataset.classes_in_dataset:
                sub_stream_data, sub_stream_label, sub_stream_idx = self.stream_dataset.get_sub_data(new_label)

                st = 0
                while True:
                    en = st + 256
                    self.swap_manager.saver.save(sub_stream_data[st:en], sub_stream_label[st:en])
                    st = en
                    if st > len(sub_stream_data):
                        break
                del sub_stream_data, sub_stream_label, sub_stream_idx


        eviction_list = multi_task_sample_update_to_RB(self.val_dataset, self.stream_dataset, True)
        
        print("\n\nEVICTION LIST  : ", eviction_list)

        print(len(self.stream_dataset))
        for idx in sorted(eviction_list, reverse=True):
            del self.stream_dataset.data[idx]
            del self.stream_dataset.targets[idx]
        print(len(self.stream_dataset))
        print("DUPLICATED DATA IN STREAM IS EVICTED\n")
        
        print("VAL SET IS UPDATED")

        
        self.cl_val_dataloader = DataLoader(self.val_dataset, batch_size=128, pin_memory=True, shuffle=True)
        self.cl_dataloader.update_loader()

        """
        if self.tasks_so_far <= 1 and (self.DP == True) and self.test_set in ["imagenet", "imagenet1000"]:
            self.model = nn.DataParallel(self.model, device_ids = self.gpu_order)
        """ 

        
        if self.old_model is not None:
            self.old_model.eval()
            print(self.device)
            self.old_model.to(self.device)
            """
            if (self.DP == True) and self.test_set in ["imagenet", "imagenet1000"]:
                self.old_model = nn.DataParallel(self.old_model, device_ids = self.gpu_order)
            """
            print("old model is available!")
            

    def after_train(self):
        self.model.eval()

        if self.swap == True:
            self.swap_manager.after_train()

        multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset)
        print("REP SET IS UPDATED")
        
        #temp acc
        print("SOFTMAX")

        
        avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc = self.eval_task(get_entropy=self.get_test_entropy)
        
        print("task_accuracy : ", task_top1_acc)
        if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
            print("task_top5_accuracy : ", task_top5_acc)

        print("current_accuracy : ", avg_top1_acc)
        if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
            print("current_top5_accuracy : ", avg_top5_acc)

        self.soft_incremental_top1_acc.append(avg_top1_acc)
        self.soft_incremental_top5_acc.append(avg_top5_acc)

        print("incremental_top1_accuracy : ", self.soft_incremental_top1_acc)
        if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
            print("incremental_top5_accuracy : ", self.soft_incremental_top5_acc)


        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        #f.write("class_accuracy : "+str(class_accuracy)+"\n")
        f.write("task_accuracy : "+str(task_top1_acc)+"\n")
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            f.write("task_top5_accuracy : "+str(task_top5_acc)+"\n")

        f.write("incremental_accuracy : "+str(self.soft_incremental_top1_acc)+"\n")
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            f.write("incremental_top5_accuracy : "+str(self.soft_incremental_top5_acc)+"\n")
        f.close()

        
        f = open(self.result_save_path + self.filename + '_time.txt','a')
        self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        f.close()
        self.curr_task_iter_time = []
        
        if self.get_loss is True:
            f = open(self.result_save_path + self.filename + '_replay_loss.txt','a')
            f.write(str(self.replay_losses)+"\n")
            f.close()

            f = open(self.result_save_path + self.filename + '_stream_loss.txt','a')
            f.write(str(self.stream_losses)+"\n")
            f.close()


        """
        if self.swap==True:
            f = open(self.result_save_path + self.filename + '_lifetime.txt', 'a')
            #print(self.how_long_stay)
            #print(self.how_much_reused)

            f.write("how long stay : "+str(self.how_long_stay)+"\n")
            f.write("how much reused : "+str(self.how_much_reused)+"\n")
            f.write(f"AVG how long stay : {np.mean(np.array(self.how_long_stay))}\n")
            f.write(f"AVG how much reused : {np.mean(np.array(self.how_much_reused))}\n")
            
            f.close()

        curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
        print("class_accuracy : ", class_accuracy)
        print("task_accuracy : ", task_accuracy)
        print("current_top1_accuracy : ", curr_top1_accuracy.item())
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            print("current_top5_accuracy : ", curr_top5_accuracy.item())
        
        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')

        self.soft_incremental_top1_acc.append(curr_top1_accuracy.item())
        self.soft_incremental_top5_acc.append(curr_top5_accuracy.item())
        print("incremental_top1_accuracy : ", self.soft_incremental_top1_acc)
        print("incremental_top5_accuracy : ", self.soft_incremental_top5_acc)

        f.write("class_accuracy : "+str(class_accuracy)+"\n")
        f.write("task_accuracy : "+str(task_accuracy)+"\n")
        f.write("incremental_accuracy : "+str(self.soft_incremental_top1_acc)+"\n")
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            f.write("incremental_top5_accuracy : "+str(self.soft_incremental_top5_acc)+"\n")
        f.close()
        
        
        f = open(self.result_save_path + self.filename + '_time.txt','a')
        self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        f.close()
        self.curr_task_iter_time = []

        """
        if self.test_set in ["imagenet", "imagenet1000"]:
            torch.save(self.model, f'./{self.filename}_task{self.tasks_so_far}.pt')

        self.old_model=copy.deepcopy(self.model)

        self.stream_dataset.clean_stream_dataset()
        gc.collect()
    
    
    def bias_forward(self, input, train=False):
        outputs_for_bias = list()
        min_class = 0

        for task_id in range(len(self.bias_layers)):
            
            max_class = max(self.data_manager.classes_per_task[task_id])
            inp_for_bias = input[:, min_class:max_class + 1]
            
            out_for_bias = self.bias_layers[task_id](inp_for_bias)

            outputs_for_bias.append(out_for_bias)
            min_class = max_class + 1
        
        return torch.cat(outputs_for_bias, dim=1)

    def train(self):
        self.model.train()

        
        if self.test_set == "cifar100":
            
            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [100,150,200], gamma=0.1)

            if self.num_epochs > 300:
                lr_change_point = list(range(100,self.num_epochs,50))
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, lr_change_point, gamma=0.2)
    
            
            self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
            self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [200,300,400], gamma=0.1)
            
            #self.bias_opt = torch.optim.Adam(params=self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.001)
            #self.bias_opt_scheduler = None

        else:
            print("IMAGENET OPTIMIZER...")

            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,  weight_decay=1e-4) #imagenet
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [30,60,80,90], gamma=0.1) #imagenet
            
            self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [30,60,80,90], gamma=0.1)


            """
            num_total_tasks = 10

            w_d = 1e-4 * (num_total_tasks/self.tasks_so_far)
            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,  weight_decay=w_d) #imagenet
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [30,60,80,90], gamma=0.1) #imagenet
            
            self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=w_d)
            self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [60,120,160,180], gamma=0.1)
            """


        for epoch in range(self.num_epochs):

            self.model.train()

            #stage 1
            print("BIAS LAYER PARAM!!!!")
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
                self.bias_layers[_].printParam(_)

            # time measure            
            iter_times = []
            iter_st = None
            swap_st,swap_en = 0,0
            stream_loss, replay_loss = [],[]

            for i, (idxs, inputs, targets) in enumerate(self.cl_dataloader):

                
                for idx in idxs:
                    if idx < len(self.replay_dataset):
                        self.num_called[idx] += 1
                self.stayed_iter = [x + 1 for x in self.stayed_iter]

            
                iter_en = time.perf_counter()
                if i > 0 and iter_st is not None:
                    iter_time = iter_en - iter_st
                    #print(f"EPOCH {epoch}, ITER {i}, ITER_TIME {iter_time} SWAP_TIME {swap_en-swap_st}...")
                    iter_times.append(iter_time)
                    if i % 10 == 0:
                        print(f"EPOCH {epoch}, ITER {i}, ITER_TIME {iter_time} SWAP_TIME {swap_en-swap_st}...")
                iter_st = time.perf_counter()


                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                outputs = self.bias_forward(outputs, train=False)                
                
                if self.swap == True and self.tasks_so_far > 1:
                    # curriculum epoch < added for early stage training
                    #if epoch >= 125 : # prof_ver. 50% gate, random
                    #if epoch < 125: # rev_prof_ver. 50% random, gate
                    #if epoch < 0: # only ver.
                    #if self.dynamic == True and epoch < (len(self.stream_dataset) * (self.tasks_so_far-1) / self.replay_size) * (1/self.swap_manager.threshold) * 5: # dynamic_ver. at least five epochs for all dataset
                    if self.dynamic == True and epoch <= int(self.num_epochs * 0.5):
                        print("random")
                        swap_idx, swap_targets = self.swap_manager.random(idxs,outputs,targets)
                        self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist())

                    else:
                        #print("swap_base")
                        swap_idx, swap_targets = self.swap_manager.swap_determine(idxs,outputs,targets)
                        swap_st = time.perf_counter()
                        self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist())
                        swap_en = time.perf_counter()

                        
                        for idx in swap_idx:
                            self.how_long_stay.append(self.stayed_iter[idx])
                            self.how_much_reused.append(self.num_called[idx])
                            
                            self.stayed_iter[idx] = 0
                            self.num_called[idx] = 0


                # BIC_swap_img : imagenet + swap + no distill 
                #if self.swap == True and self.test_set in ["imagenet", "imagenet1000"]:
                #    print("THIS IS IMAGENET SWAP VER")
                #    loss = nn.CrossEntropyLoss()(outputs, targets)

                    
                if self.distill == False:
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                
                elif self.old_model is not None:
                    T = 2
                    with torch.no_grad():
                        old_outputs = self.old_model(inputs)
                        old_outputs = self.bias_forward(old_outputs, train=False)
                        old_task_size = old_outputs.shape[1]
                    
                        old_logits = old_outputs.detach()

                    hat_pai_k = F.softmax(old_logits/T, dim=1)
                    log_pai_k = F.log_softmax(outputs[..., :old_task_size]/T, dim=1)

                    loss_soft_target = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                    loss_hard_target = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
                    
                    """
                    old_outputs = F.softmax(old_outputs/T, dim=1)
                    old_task_size = old_outputs.shape[1]
                
                    log_outputs = F.log_softmax(outputs[..., :old_task_size]/T, dim=1)

                    loss_soft_target = -torch.mean(torch.sum(old_outputs * log_outputs, dim=1))
                    loss_hard_target = nn.CrossEntropyLoss()(outputs, targets)
                    """

                    
                    if self.get_loss == True:
                        get_loss = loss_hard_target.clone().detach()
                        #get_loss = loss_ext.view(loss_ext.size(0), -1)
                        #get_loss = loss_ext.mean(-1)
                        replay_idxs = (idxs < self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        stream_idxs = (idxs >= self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        stream_loss.append(get_loss[stream_idxs].mean(-1).item())
                        if get_loss[replay_idxs].size(0) > 0:
                            replay_loss.append(get_loss[replay_idxs].mean(-1).item())
                        
                    loss_hard_target = loss_hard_target.mean()

                    if self.distill == False:
                        self.alpha = 0
                        #print(f"No distill ... alpha will be {self.alpha}")
                        
                    loss = (self.alpha * loss_soft_target) + ((1-self.alpha) * loss_hard_target)
                    #loss = (loss_soft_target * T * T) + ((1-self.alpha) * loss_hard_target)
                    
                    
                else:
                    loss = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
                    #print(loss.shape)
                    if self.get_loss == True:
                        get_loss = loss.clone().detach()
                        #get_loss = loss_ext.view(loss_ext.size(0), -1)
                        #get_loss = loss_ext.mean(-1)
                        replay_idxs = (idxs < self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        stream_idxs = (idxs >= self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        #print(get_loss)
                        #print(stream_idxs)
                        stream_loss.append(get_loss[stream_idxs].mean(-1).item())
                        replay_loss.append(get_loss[replay_idxs].mean(-1).item())
                        
                    loss = loss.mean()

                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
            print("lr {}".format(self.opt.param_groups[0]['lr']))
            print("epoch {}, loss {}".format(epoch, loss.item()))
            self.loss_item.append(loss.item())
            self.stream_losses.append(np.mean(np.array(stream_loss)))
            self.replay_losses.append(np.mean(np.array(replay_loss)))


            if epoch > 0:
                
                #print(iter_times)
                self.curr_task_iter_time.append(np.mean(np.array(iter_times)))
            
            if epoch % 10 == 0 and epoch > 0:
                
                avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc = self.eval_task()
                print("\n\n============ ACC - INTERMEDIATE ============")
                print("task_accuracy : ", task_top1_acc)
                print("task_top5_accuracy : ", task_top5_acc)

                print("current_accuracy : ", avg_top1_acc)
                print("current_top5_accuracy : ", avg_top5_acc)
                print("============================================\n\n")
                """
                curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
                print("\n\n============ ACC - INTERMEDIATE ============")
                print("soft_class_accuracy : ", class_accuracy)
                print("soft_task_accuracy : ", task_accuracy)
                print("soft_current_top1_accuracy : ", curr_top1_accuracy.item())
                print("soft_current_top5_accuracy : ", curr_top5_accuracy.item())
                print("============================================\n\n")
                """
            #self.num_swap.append(self.swap_manager.get_num_swap())
            #self.swap_manager.reset_num_swap()
            
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        #stage 2
        if self.old_model is not None:

            print("Training bias layers....")

            print("\nBEFORE BIAS TRAINING : BIAS LAYER PARAM!!!!")

            
            for _ in range(len(self.bias_layers)-1):
                self.bias_layers[_].eval()
                #self.bias_layers[_].train()
                self.bias_layers[_].printParam(_)
            
            self.bias_layers[len(self.bias_layers)-1].train()
            self.bias_layers[len(self.bias_layers)-1].printParam(len(self.bias_layers)-1)
            print("\n")
            
            #for epoch in range(2 * self.num_epochs):
            for epoch in range(self.num_epochs):    
                print(f"Training bias layers....epoch {epoch}")

                for i, (idxs, inputs, targets) in enumerate(self.cl_val_dataloader):
                    self.model.eval()

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(inputs)
                    outputs = self.bias_forward(outputs, train=True)

                    loss = self.criterion(outputs, targets)
                    self.bias_opt.zero_grad()
                    loss.backward()
                    self.bias_opt.step()
                if self.bias_opt_scheduler is not None:
                    self.bias_opt_scheduler.step()
                print("lr {}".format(self.bias_opt.param_groups[0]['lr']))

                if epoch % 100 == 0 and epoch > 0:
                    avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc = self.eval_task()
                    print("\n\n============ ACC - INTERMEDIATE ============")
                    print("task_accuracy : ", task_top1_acc)
                    print("task_top5_accuracy : ", task_top5_acc)

                    print("current_accuracy : ", avg_top1_acc)
                    print("current_top5_accuracy : ", avg_top5_acc)
                    print("============================================\n\n")
                    """    
                    curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
                    print("\n\n============ ACC - INTERMEDIATE ============")
                    print("class_accuracy : ", class_accuracy)
                    print("task_accuracy : ", task_accuracy)
                    print("current_top1_accuracy : ", curr_top1_accuracy.item())
                    print("current_top5_accuracy : ", curr_top5_accuracy.item())
                    print("============================================\n\n")
                    """
                    for _ in range(len(self.bias_layers)-1):
                        self.bias_layers[_].eval()
                        #self.bias_layers[_].train()
                        self.bias_layers[_].printParam(_)
                    
                    self.bias_layers[len(self.bias_layers)-1].train()
                    self.bias_layers[len(self.bias_layers)-1].printParam(len(self.bias_layers)-1)
                    print("\n")

            
            print("\nAFTER BIAS TRAINING : BIAS LAYER PARAM!!!!")
            for _ in range(len(self.bias_layers)-1):
                self.bias_layers[_].printParam(_)
            self.bias_layers[len(self.bias_layers)-1].printParam(len(self.bias_layers)-1)
            print("\n")



    def eval_task(self, get_entropy=False):
        self.model.eval()
        
        for _ in range(len(self.bias_layers)):
           self.bias_layers[_].eval()
        
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        ypreds, ytrue = self.compute_accuracy(test_dataloader, get_entropy)

        
        avg_top1_acc, task_top1_acc = self.accuracy_per_task(ypreds, ytrue, task_size=10, topk=1)
        avg_top5_acc, task_top5_acc = self.accuracy_per_task(ypreds, ytrue, task_size=10, topk=5)

        return avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc

    
    def compute_accuracy(self, loader, get_entropy=False):
        ypred, ytrue = [], []

        
        if self.swap==True and get_entropy == True:
            w_entropy_test = []
            r_entropy_test = []

            logits_list = []
            labels_list = []

        for setp, ( imgs, labels) in enumerate(loader):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = self.bias_forward(outputs)
            outputs = outputs.detach()
            #get entropy of testset
            if self.swap==True and get_entropy == True:
                r, w = self.get_entropy(outputs, labels)
                r_entropy_test.extend(r)
                w_entropy_test.extend(w)
                    
                logits_list.append(outputs)
                labels_list.append(labels)


            ytrue.append(labels.numpy())
            ypred.append(torch.softmax(outputs, dim=1).cpu().numpy())

        if self.swap==True and get_entropy == True:
            print("RECORD TEST ENTROPY!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            f = open(self.result_save_path + self.filename + '_correct_test_entropy.txt', 'a')
            f.write(str(r_entropy_test)+"\n")
            f.close()
            
            f = open(self.result_save_path + self.filename + '_wrong_test_entropy.txt', 'a')
            f.write(str(r_entropy_test)+"\n")
            f.close()

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

            ece = self.ece_loss(logits, labels).item()
            f = open(self.result_save_path + self.filename + '_ece_test.txt', 'a')
            f.write(str(ece)+"\n")
            f.close()

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue


    def accuracy_per_task(self,ypreds, ytrue, task_size=10, topk=1):
        """Computes accuracy for the whole test & per task.
        :param ypred: The predictions array.
        :param ytrue: The ground-truth array.
        :param task_size: The size of the task.
        :return: A dictionnary.
        """
        all_acc = {}

        avg_acc = self.accuracy(ypreds, ytrue, topk=topk) * 100
        
        task_acc = {}

        if task_size is not None:
            for task_id, class_id in enumerate(range(0, np.max(ytrue) + task_size, task_size)):
                if class_id > np.max(ytrue):
                    break

                idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

                label = "{}-{}".format(
                    str(class_id).rjust(2, "0"),
                    str(class_id + task_size - 1).rjust(2, "0")
                )
                #all_acc[label] = self.accuracy(ypreds[idxes], ytrue[idxes], topk=topk)
                task_acc[task_id] = self.accuracy(ypreds[idxes], ytrue[idxes], topk=topk) * 100

        return avg_acc, task_acc

    def accuracy(self,output, targets, topk=1):
        """Computes the precision@k for the specified values of k"""
        output, targets = torch.tensor(output), torch.tensor(targets)

        batch_size = targets.shape[0]
        if batch_size == 0:
            return 0.
        nb_classes = len(np.unique(targets))
        topk = min(topk, nb_classes)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].reshape(-1).float().sum(0).item()
        return round(correct_k / batch_size, 4)



    """

    def eval(self, mode=1):
        
        self.model.eval()
        
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        
        if mode==0:
            print("compute NMS")
            
        self.model.eval()
        for _ in range(len(self.bias_layers)):
           self.bias_layers[_].eval()
        
        correct, total = 0, 0
        class_correct = list(0. for i in range(self.classes_so_far))
        class_total = list(0. for i in range(self.classes_so_far))
        class_accuracy = list()
        top5_accuracy = list()
        task_accuracy = dict()

        for setp, ( imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
                outputs = self.bias_forward(outputs)

            #top5 acc
            top5_acc = self.top5_acc(outputs, labels)
            top5_accuracy.append(top5_acc.item())

            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            c = (predicts.cpu() == labels.cpu()).squeeze()

            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
        for i in range(len(class_correct)):
            if class_correct[i]==0 and class_total[i] == 0:
                continue
            class_acc = 100 * class_correct[i] / class_total[i]
            print('[%2d] Accuracy of %2d : %2d %%' % (
            i, i, class_acc))
            class_accuracy.append(class_acc)
        
        for task_id, task_classes in self.data_manager.classes_per_task.items():
            task_acc = np.mean(np.array(list(map(lambda x : class_accuracy[x] ,task_classes))))
            task_accuracy[task_id] = task_acc

        total_top1_accuracy = 100 * correct / total
        total_top5_accuracy = np.mean(np.array(top5_accuracy))
        
        self.model.train()
        return total_top1_accuracy, total_top5_accuracy, task_accuracy, class_accuracy
    
    def top5_acc(self, output, target):
        with torch.no_grad():
            batch_size = target.size(0)

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
            res = correct_k.mul_(100.0 / batch_size)
            
            return res

    def classify(self, test_image):
        result = []
        test_image = F.normalize(self.model.feature_extractor(test_image).detach()).cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        #print(class_mean_set)

        for target in test_image:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
    """