#https://github.com/mmasana/FACIL/blob/master/src/approach/bic.py

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
from _utils.sampling import multi_task_sample_update_to_RB
from lib import utils

import math
import time

from dataset.validate import ValidateDataset

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
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    self.nomalize,
                                                ])

        else:
            val_transform = transforms.Compose([transforms.ToTensor(),
                                            self.nomalize])

        self.classes_so_far = 0
        self.tasks_so_far = 0

        self.soft_incremental_top1_acc = list()
        self.soft_incremental_top5_acc = list()
        
        self.num_swap = list()
        self.bias_layers = list()

        self.criterion = nn.CrossEntropyLoss()

        self.old_model = None
        #
        # add validation set
        #
        self.val_dataset = ValidateDataset(int(self.replay_dataset.rb_size * 0.1), val_transform, sampling="ringbuffer")
        
        self.replay_dataset.rb_size = int(self.replay_dataset.rb_size * 0.9)

        if 'store_ratio' in kwargs:
            self.store_ratio = float(kwargs['store_ratio'])
            print("STORE RATIO : ", self.store_ratio)
        else:
            self.store_ratio = 1
        
        if 'gpu_order' in kwargs:
            self.gpu_order = kwargs['gpu_order']
            print("GPU ORDER : ", self.gpu_order)
        else:
            self.gpu_order = torch.cuda.device_count()

    def before_train(self, task_id):
        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)

        self.alpha = self.classes_so_far / (self.classes_so_far + len(self.stream_dataset.classes_in_dataset))
        
        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        print("classes_so_far : ", self.classes_so_far)

        self.tasks_so_far += 1
        print("tasks_so_far : ", self.tasks_so_far)
        
        if self.tasks_so_far > 1 and self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            self.model.module.Incremental_learning(self.classes_so_far)

        else:
            self.model.Incremental_learning(self.classes_so_far)        
        
        self.model.train()
        self.model.to(self.device)

        print("length of replay dataset : ", len(self.replay_dataset))
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
                print(len(sub_stream_data), self.store_ratio)
                st = 0
                while True:
                    en = st + 20
                    self.swap_manager.saver.save(sub_stream_data[st:en], sub_stream_label[st:en])
                    st = en
                    if st > math.ceil(len(sub_stream_data) * self.store_ratio):
                        break
                print("How much data will be saved for per class ? ", math.ceil(len(sub_stream_data) * self.store_ratio))
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

        
        if self.tasks_so_far <= 1 and self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_order)
            

        if self.old_model is not None:
            self.old_model.eval()
            print(self.device)
            self.old_model.to(self.device)
            if self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
                self.old_model = nn.DataParallel(self.old_model, device_ids=self.gpu_order)
            print("old model is available!")
            

    def after_train(self):
        self.model.eval()

        if self.swap == True:
            self.swap_manager.after_train()

        multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset)
        print("REP SET IS UPDATED")
        
        #temp acc
        print("SOFTMAX")
        curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
        print("class_accuracy : ", class_accuracy)
        print("task_accuracy : ", task_accuracy)
        print("current_top1_accuracy : ", curr_top1_accuracy.item())
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            print("current_top5_accuracy : ", curr_top5_accuracy.item())
        
        f = open('results_exp/'+self.filename+'.txt', 'a')
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
        
        f = open('num_swap/'+self.filename+'.txt', 'a')
        f.write(str(self.num_swap)+"\n")
        f.close()
        
        if self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            self.old_model=copy.deepcopy(self.model.module)
        else:
            self.old_model=copy.deepcopy(self.model)

        self.stream_dataset.clean_stream_dataset()
    
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
    
    
    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def train(self):
        self.model.train()

        
        if self.test_set == "cifar100":
            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [100,150,200], gamma=0.1)
            
            #self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
            #self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [200,300,400], gamma=0.1)

        else:
            print("IMAGENET OPTIMIZER...")
            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,  weight_decay=0.0001) #imagenet
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [30,60,80,90], gamma=0.1) #imagenet
            
            #self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
            #self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [60,120,160,180], gamma=0.1)
        
        self.bias_opt = torch.optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.05, momentum=0.9)


        times = []
        time_l = []
        st = None

        for epoch in range(self.num_epochs):
            en = time.perf_counter()
            if st is not None:
                time_l.append(en-st)
                print("\n\nTIME {}".format(np.mean(np.array(time_l))))
                if len(time_l) > 10:
                #    print("\n\nTIME {}".format(np.mean(np.array(time_l))))
                    times.append(np.mean(np.array(time_l)))
                    time_l =[]
            st = time.perf_counter()

            self.model.train()

            #stage 1
            print("BIAS LAYER PARAM!!!!")
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
                self.bias_layers[_].printParam(_)

            iter_st = None
            for i, (idxs, inputs, targets) in enumerate(self.cl_dataloader):
                iter_en = time.perf_counter()
                if iter_st is not None and i % 20 == 0:
                    print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")
                iter_st = time.perf_counter()

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                #with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                outputs = self.bias_forward(outputs, train=False)
                
            
                if self.swap == True and self.tasks_so_far > 1:
                    swap_idx, swap_targets = self.swap_manager.swap_determine(idxs,outputs,targets)
                    self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist())
                
                # BIC_swap_img : imagenet + swap + no distill 
                if self.swap == True and self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
                    loss = nn.CrossEntropyLoss()(outputs, targets)

                elif self.old_model is not None:
                    T = 2
                    with torch.no_grad():
                        targets_old = self.old_model(inputs)
                        targets_old = self.bias_forward(targets_old, train=False)

                    old_task_size = targets_old.shape[1]

                    loss_soft_target = self.cross_entropy(outputs[..., :old_task_size], 
                                            targets_old, exp=1.0 / 2.0)

                    loss_hard_target = nn.functional.cross_entropy(outputs, targets)
                    loss = (self.alpha * loss_soft_target) + ((1-self.alpha) * loss_hard_target)

                    
                else:
                    loss = nn.functional.cross_entropy(outputs, targets)
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                #
                #self.opt.zero_grad()
                #self.scaler.scale(loss).backward()
                #self.scaler.step(self.opt)
                #self.scaler.update()
                
            

            print("lr {}".format(self.opt.param_groups[0]['lr']))
            print("epoch {}, loss {}".format(epoch, loss.item()))
            self.loss_item.append(loss.item())

            if epoch % 10 == 0 and epoch > 0:
                curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
                print("\n\n============ ACC - INTERMEDIATE ============")
                print("soft_class_accuracy : ", class_accuracy)
                print("soft_task_accuracy : ", task_accuracy)
                print("soft_current_top1_accuracy : ", curr_top1_accuracy.item())
                print("soft_current_top5_accuracy : ", curr_top5_accuracy.item())
                print("============================================\n\n")
            
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
            
            self.bias_epochs = 200
            for epoch in range(self.bias_epochs):
                
                print(f"Bias training .. {epoch}")
                for i, (idxs, inputs, targets) in enumerate(self.cl_val_dataloader):
                    self.model.eval()

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(inputs)
                    outputs = self.bias_forward(outputs, train=True)

                    loss = nn.functional.cross_entropy(outputs, targets)
                    loss += 0.1 * ((self.bias_layers[len(self.bias_layers)-1].beta[0] ** 2) / 2)

                    self.bias_opt.zero_grad()
                    loss.backward()
                    self.bias_opt.step()
                #self.bias_opt_scheduler.step()
                print("lr {}".format(self.bias_opt.param_groups[0]['lr']))

                if epoch % 100 == 0 and epoch > 0:
                    curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
                    print("\n\n============ ACC - INTERMEDIATE ============")
                    print("class_accuracy : ", class_accuracy)
                    print("task_accuracy : ", task_accuracy)
                    print("current_top1_accuracy : ", curr_top1_accuracy.item())
                    print("current_top5_accuracy : ", curr_top5_accuracy.item())
                    print("============================================\n\n")

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
