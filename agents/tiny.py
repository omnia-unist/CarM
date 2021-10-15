from random import random
from torch.nn import functional as F
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import copy
from collections import OrderedDict 

import csv
import os
from agents.base import Base
from _utils.sampling import multi_task_sample_update_to_RB
from lib import utils
from scipy.spatial.distance import cdist
from dataset.dataloader import AserDataLoader, TinyReplayDataLoader
import time
from dataset.test import get_test_set


class Tiny(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)
        self.cl_dataloader = AserDataLoader(stream_dataset, replay_dataset, data_manager, 0, 10, swap)

        # variables 
        self.num_swap = list()
        self.task_number = 0
        self.samples_so_far = 0

        # variables for evaluation
        self.classes_so_far = 0
        self.tasks_so_far = 0
        self.incremental_acc = list()
        self.epoch_acc = list()
        self.max_epoch_acc = 0
        self.soft_class_incremental_acc = list()
        self.soft_task_incremental_acc = list()

        # specify the start and ending learning rates and weight decay to be used
        self.maxlr = 0.1
        self.minlr = 0.0005
        self.weight_decay = 0

        # optimizer, scheduler, and criterion initialization
        self.criterion = torch.nn.CrossEntropyLoss(reduce = False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.maxlr, weight_decay=self.weight_decay)

    def before_train(self, task_id):
        self.curr_task_iter_time = []

        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)


        self.replay_dataloader = TinyReplayDataLoader(self.replay_dataset, self.data_manager, self.cl_dataloader.num_workers, self.cl_dataloader.batch_size)
        self.iter_r = iter(self.replay_dataloader)


        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        self.tasks_so_far += 1
        print("classes_so_far : ", self.classes_so_far)
        self.model.Incremental_learning(self.classes_so_far)
        self.model.train()
        self.model.to(self.device)
        print("length of replay dataset : ", len(self.replay_dataset))
        # store images from stream to hard drive....
        if self.swap is True:
            self.swap_manager.before_train()
            
        self.cl_dataloader.update_loader()
        
        self.stream_losses, self.replay_losses = list(), list()

        
    def train(self):
        # number of iterations through the data
        for epoch in range(self.num_epochs):
            # iterate batch wise through the stream data using the data loader
            
            # time measure
            iter_times = []
            iter_st = None
            stream_loss, replay_loss = [],[]

            for data_stream_count, (stream_idxs, inputs, targets) in enumerate(self.cl_dataloader):
                iter_en = time.perf_counter()
                if data_stream_count > 0 and iter_st is not None:
                    iter_time = iter_en - iter_st
                    
                    iter_times.append(iter_time)
                    if data_stream_count % 20 == 0:
                        print(f"EPOCH {epoch}, ITER {data_stream_count}, TIME {iter_en-iter_st}...")
                iter_st = time.perf_counter()


                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # retrieve samples
                #retrieved_idxs, mem_inputs, mem_targets = self.random_retrieve(inputs, targets)

                if len(self.replay_dataset)>0:
                    retrieved_idxs, mem_inputs, mem_targets, replay_ids = next(self.iter_r)
                    
                    mem_inputs = mem_inputs.to(self.device)
                    mem_targets = mem_targets.to(self.device)

                    # combine the stream and memory samples
                    combined_inputs = torch.cat((mem_inputs, inputs))
                    combined_targets = torch.cat((mem_targets, targets))
                else:
                    combined_inputs = inputs
                    combined_targets = targets
                # task incremental training
                combined_logits = self.model.forward(combined_inputs)
                
                if self.swap == True and len(self.replay_dataset)>0:
                    #print(retrieved_idxs.shape, combined_logits.shape, mem_targets.shape)
                    swap_idx, swap_targets, swap_ids = self.swap_manager.swap_determine(retrieved_idxs, combined_logits[:mem_targets.shape[0], ...], mem_targets, replay_ids)
                    self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist(), swap_ids.tolist())

                combined_loss = []
                for current_task in range(self.tasks_so_far):
                    # copy the logits for modification
                    temp_logits = torch.clone(combined_logits)
                    # the task numbers which should not be changed
                    do_not_modify = self.data_manager.classes_per_task[current_task]
                    # modify logits of 'out of task' to infinity
                    for logits_index in range(len(temp_logits)):
                        indices = np.arange(len(temp_logits[logits_index]))
                        indices_to_modify = [x for x in indices if x not in do_not_modify]
                        temp_logits[logits_index][indices_to_modify] = float('-inf')
                    # run the cross entropy using the modified logits
                    temp_loss = self.criterion(temp_logits, combined_targets)
                    # if the target of the sample is within the task append the cross entropy of that sample


                    if self.get_loss == True:
                        get_loss = temp_loss.clone().detach()
                        stream_loss.append(get_loss[:10].mean(-1).item())
                        if len(get_loss) > 10:
                            replay_loss.append(get_loss[-10:].mean(-1).item())


                    for target_index in range(len(combined_targets)):
                        if combined_targets[target_index] in do_not_modify:
                            combined_loss.append(temp_loss[target_index])

                combined_loss = torch.sum(torch.stack(combined_loss).to(self.device)) / len(combined_targets)

                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()
                # update the memory
                self.reservoir_update(data_stream_count, stream_idxs, inputs, targets)
                self.samples_so_far += 10
                # swap is done here, swap only if memory is full
                if self.swap == True:
                    stream_data_batch = [self.stream_dataset.data[idx] for idx in stream_idxs]
                    self.swap_manager.saver.save(stream_data_batch, targets.tolist())
            
            print(iter_times)
            self.curr_task_iter_time.append(np.mean(np.array(iter_times)))

            self.stream_losses.append(np.mean(np.array(stream_loss)))
            self.replay_losses.append(np.mean(np.array(replay_loss)))
            
            if self.swap == True:
                print("epoch {}, loss {}, num_swap {}".format(epoch, combined_loss.item(), self.swap_manager.get_num_swap()))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()

                self.swap_manager.swap_class_dist = dict(sorted(self.swap_manager.swap_class_dist.items()))
                f = open(self.result_save_path + self.filename + '_distribution.txt', 'a')
                f.write("incremental_accuracy : "+str(self.swap_manager.swap_class_dist)+"\n")
                f.close()
                self.swap_manager.reset_swap_class_dist()

            

    def after_train(self):
        if self.swap == True:
            self.swap_manager.after_train()
            
            self.swap_manager.reset_num_swap()

        self.stream_dataset.clean_stream_dataset()
        self.task_number += 1
        # write stuff
        
        curr_accuracy, task_accuracy, class_accuracy = self.class_eval()    
        print("c_class_accuracy : ", class_accuracy)
        print("c_task_accuracy : ", task_accuracy)
        print("c_current_accuracy : ", curr_accuracy.item())
        
        self.soft_class_incremental_acc.append(curr_accuracy.item())
        print("c_incremental_accuracy : ", self.soft_class_incremental_acc)

        curr_accuracy, task_accuracy, class_accuracy = self.task_eval(get_entropy=self.get_test_entropy)    
        print("t_class_accuracy : ", class_accuracy)
        print("t_task_accuracy : ", task_accuracy)
        print("t_current_accuracy : ", curr_accuracy.item())
        
        self.soft_task_incremental_acc.append(curr_accuracy.item())
        print("t_incremental_accuracy : ", self.soft_task_incremental_acc)

        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        f.write("class_accuracy : "+str(class_accuracy)+"\n")
        f.write("task_accuracy : "+str(task_accuracy)+"\n")
        f.write("class_incremental_accuracy : "+str(self.soft_class_incremental_acc)+"\n")
        f.write("task_incremental_accuracy : "+str(self.soft_task_incremental_acc)+"\n")
        f.close()

        
        f = open(self.result_save_path + self.filename + '_time.txt','a')
        self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        f.close()

        
        
        if self.swap==True:
            f = open(self.result_save_path + self.filename + f'_num_swap_{self.swap_base}.csv','a',newline="")
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.num_swap)
            f.close()
            self.num_swap = []


        print('======================================================================')

    def task_eval(self, get_entropy=False):
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        self.model.eval()
        correct, total = 0, 0
        class_correct = list(0. for i in range(self.classes_so_far))
        class_total = list(0. for i in range(self.classes_so_far))
        class_accuracy = list()
        task_accuracy = dict()
        
        if self.swap==True and get_entropy == True:
            w_entropy_test = []
            r_entropy_test = []

            logits_list = []
            labels_list = []


        for setp, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
            # for the task of the output prune the other task outputs
            for logits_index in range(len(logits)):
                # get the task of the target class
                for task_id, classes_of_task in self.data_manager.classes_per_task.items():
                    if targets[logits_index] in classes_of_task:
                        do_not_modify = self.data_manager.classes_per_task[task_id]
                        break
                # modify logits of 'out of task' to infinity
                indices = np.arange(len(logits[logits_index]))
                indices_to_modify = [x for x in indices if x not in do_not_modify]
                logits[logits_index][indices_to_modify] = float('-inf')
            # other stuff are the same
            predicts = torch.max(logits, dim=1)[1]
            c = (predicts.cpu() == targets.cpu()).squeeze()
            correct += (predicts.cpu() == targets.cpu()).sum()
            total += len(targets)


            if self.swap==True and get_entropy == True:
                r, w = self.get_entropy(logits, targets)
                r_entropy_test.extend(r)
                w_entropy_test.extend(w)
                
                logits_list.append(logits)
                labels_list.append(targets)


            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


        if self.swap==True and get_entropy == True:
            print("RECORD TEST ENTROPY!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            f = open(self.result_save_path + self.filename + '_correct_test_entropy.txt', 'a')
            f.write(str(r_entropy_test)+"\n")
            f.close()
            
            f = open(self.result_save_path + self.filename + '_wrong_test_entropy.txt', 'a')
            f.write(str(r_entropy_test)+"\n")
            f.close()

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

            ece = self.ece_loss(logits, labels).item()
            f = open(self.result_save_path + self.filename + '_ece_test.txt', 'a')
            f.write(str(ece)+"\n")
            f.close()

                
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
        total_accuracy = 100 * correct / total
        self.model.train()
        return total_accuracy, task_accuracy, class_accuracy

    def class_eval(self):
        self.model.eval()
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        
        correct, total = 0, 0
        class_correct = list(0. for i in range(self.classes_so_far))
        class_total = list(0. for i in range(self.classes_so_far))
        class_accuracy = list()
        task_accuracy = dict()

        for setp, ( imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs) 
            predicts = torch.max(outputs, dim=1)[1] 
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

        total_accuracy = 100 * correct / total
        self.model.train()
        return total_accuracy, task_accuracy, class_accuracy

    def reservoir_update(self, data_stream_count, stream_idx, inputs, targets):
        batch_count = 0
        for index, new_data, new_label in zip(stream_idx, inputs, targets):
            if(len(self.replay_dataset.data) < self.replay_dataset.rb_size):
                self.replay_dataset.data.append(self.stream_dataset.data[index])
                self.replay_dataset.targets.append(new_label.item())
                self.replay_dataset.tracker.append(self.data_manager.num_samples_observed_so_far)
            else:
                random_index = np.random.randint(0, self.samples_so_far + batch_count)
                if random_index < self.replay_dataset.rb_size:
                    self.replay_dataset.data[random_index] = self.stream_dataset.data[index]
                    self.replay_dataset.targets[random_index] = new_label.item()
                    self.replay_dataset.tracker[random_index] = self.data_manager.num_samples_observed_so_far
            batch_count += 1
            self.data_manager.increase_observed_samples()
        return

    def ring_buffer(self, data_stream_count, stream_idx, inputs, targets):
        selected_indices = []
        # space available in the memory
        space_left = self.replay_dataset.rb_size - len(self.replay_dataset.data)
        # remove random samples from memory if the space avaiable is not enough for the batch
        # get the indices of memory samples
        if(space_left < len(inputs)):
            filled_indices = np.arange(len(self.replay_dataset.data))
            selected_indices = np.random.choice(filled_indices, len(inputs) - space_left, replace=False)
        # arrange the indices so that it wont affect later evictions
        selected_indices = sorted(selected_indices, reverse = True)
        # eviction part
        for mem_index in selected_indices:
            self.replay_dataset.len_per_cls[self.replay_dataset.targets[mem_index]] -= 1
            self.offset_adjust(self.replay_dataset, adjust_insert = 0, label = self.replay_dataset.targets[mem_index])
            if self.replay_dataset.len_per_cls[self.replay_dataset.targets[mem_index]] <= 0:
                del self.replay_dataset.len_per_cls[self.replay_dataset.targets[mem_index]]
                del self.replay_dataset.offset[self.replay_dataset.targets[mem_index]]
            self.replay_dataset.data.pop(mem_index)
            self.replay_dataset.targets.pop(mem_index)
        # insertion part
        for index, new_data, new_label in zip(stream_idx, inputs, targets):
            # if the label is not in memory, make records of set and len
            if new_label.item() not in self.replay_dataset.offset:
                self.replay_dataset.len_per_cls[new_label.item()] = 0
                # new entry in offset
                if len(self.replay_dataset) > 0:
                    i = 1
                    while (1):
                        if (new_label.item() - i) in self.replay_dataset.offset:
                            self.replay_dataset.offset[new_label.item()] = self.replay_dataset.offset[new_label.item() - i] + self.replay_dataset.len_per_cls[new_label.item() - i]
                            break
                        if (new_label.item() - i) < 0:
                            self.replay_dataset.offset[new_label.item()] = 0
                            break
                        i += 1
                else:
                    self.replay_dataset.offset[new_label.item()] = 0
            # add the new_data to the memory
            self.replay_dataset.data.insert(self.replay_dataset.offset[new_label.item()] + self.replay_dataset.len_per_cls[new_label.item()], self.stream_dataset.data[index])
            self.replay_dataset.targets.insert(self.replay_dataset.offset[new_label.item()] + self.replay_dataset.len_per_cls[new_label.item()], new_label.item())
            self.replay_dataset.len_per_cls[new_label.item()] += 1
            self.offset_adjust(self.replay_dataset, adjust_insert = 1, label = new_label.item())
        print('Filled randomly, memory size: ' + str(len(self.replay_dataset.data)))
        return

    def random_retrieve(self, inputs, targets):
        retrieved_x, retrieved_y = [], []
        # if the buffer is empty, do nothing
        if len(self.replay_dataset.data) == 0:
            selected_indices, retrieved_x, retrieved_y = list(), inputs, targets
        else:
            filled_indices = np.arange(len(self.replay_dataset.data))
            selected_indices = np.random.choice(filled_indices, len(inputs), replace=False)
            for entry in selected_indices:
                idx, temp_x, temp_y = self.replay_dataset[entry]
                retrieved_x.append(temp_x)
                retrieved_y.append(temp_y)
            # convert to tensors
            retrieved_x = torch.stack(retrieved_x).to(self.device)
            retrieved_y = torch.tensor(retrieved_y).to(self.device)
        return selected_indices, retrieved_x, retrieved_y

    def offset_adjust(self, dataset, adjust_insert = 1, label = None):
        # change the offset of other classes when samples are added or deleted
        if(adjust_insert):
            # increase offsets of proceeding classses
            for i in range(label + 1, max(self.replay_dataset.offset) + 1):
                if i in self.replay_dataset.offset:
                    self.replay_dataset.offset[i] += 1
        else:
            # decrease offsets of proceeding classses
            for i in range(label + 1, max(self.replay_dataset.offset) + 1):
                if i in self.replay_dataset.offset:
                    self.replay_dataset.offset[i] -= 1
        return
