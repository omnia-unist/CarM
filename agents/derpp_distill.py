from torch.nn import functional as F
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import copy
from array import array

import os
from agents.base import Base
from _utils.sampling import sample_update_to_RB
from lib import utils
import time
import gc

import csv
from dataset.dataloader import TinyContinualDataLoader, TinyReplayDataLoader

class DarkERPlus_distill(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)
        
        self.replay_dataset.sampling = "reservoir_der"

        self.classes_so_far = 0
        self.tasks_so_far = 0
        
        self.nmc_incremental_acc = list()
        self.soft_incremental_acc = list()
        self.num_swap = list()
        #self.cl_dataloader = DERContinualDataLoader(self.stream_dataset, self.replay_dataset, self.data_manager, 
        #                                            self.cl_dataloader.num_workers, self.cl_dataloader.batch_size, self.swap)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.model.parameters(), lr=self.lr)
        
        if 'alpha' in kwargs:
            self.alpha_set = float(kwargs['alpha'])
        else:
            self.alpha_set = -1

        #self.beta = kwargs['beta']


    def before_train(self, task_id):
        self.model.eval()
        
        self.curr_task_iter_time = []
        
        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)

        self.alpha = self.classes_so_far / (self.classes_so_far + len(self.stream_dataset.classes_in_dataset))


        self.cl_dataloader = TinyContinualDataLoader(self.stream_dataset, self.data_manager, 
                                                    self.cl_dataloader.num_workers, self.cl_dataloader.batch_size, self.swap)
        self.replay_dataloader = TinyReplayDataLoader(self.replay_dataset, self.data_manager, self.cl_dataloader.num_workers, self.cl_dataloader.batch_size)
        self.iter_r = iter(self.replay_dataloader)

        #self.cl_dataloader.update_loader()

        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        print("classes_so_far : ", self.classes_so_far)

        self.tasks_so_far += 1
        print("tasks_so_far : ", self.tasks_so_far)

        #self.model.Incremental_learning(self.classes_so_far)
        self.model.train()
        self.model.to(self.device)

        #print("length of replay dataset : ", len(self.replay_dataset))

        if self.swap is True:
            self.swap_manager.before_train()

    def after_train(self):

        self.img_sizes = []
        
        import sys
        for inp in self.replay_dataset.data:
            #print(inp.tobytes())
            img_size = sys.getsizeof(inp.tobytes())
            self.img_sizes.append(img_size)
        print("=================================\n")
        #print(self.img_sizes)
        print("AVERAGE IMAGE SIZE : ", np.array(np.mean(self.img_sizes)))
        print("=================================\n")


        self.model.eval()
        
        f = open(self.result_save_path + self.filename + '_loss.txt', 'a')
        f.write(str(self.loss_item)+"\n")
        self.loss_item = []
        f.close()
        
        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        curr_accuracy, task_accuracy, class_accuracy = self.eval(get_entropy=self.get_test_entropy)
        print("class_accuracy : ", class_accuracy)
        print("task_accuracy : ", task_accuracy)
        print("current_accuracy : ", curr_accuracy.item())
        
        self.soft_incremental_acc.append(curr_accuracy.item())
        print("incremental_accuracy : ", self.soft_incremental_acc)

        f.write("class_accuracy : "+str(class_accuracy)+"\n")
        f.write("task_accuracy : "+str(task_accuracy)+"\n")
        f.write("incremental_accuracy : "+str(self.soft_incremental_acc)+"\n")
        f.close()

        
        f = open(self.result_save_path + self.filename + '_time.txt','a')
        self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        f.close()

        
        if self.swap is True:
            self.swap_manager.after_train()


        self.stream_dataset.clean_stream_dataset()
        gc.collect()

    def expend_dim(self, logits, n_classes):
        zero_tensor = torch.zeros(n_classes-logits.shape[0]).to(logits.device)
        logits = torch.cat((logits, zero_tensor))

        return logits

    
    def train(self):
        print("TRAIN ID : ", os.getpid())
        self.model.train()

        total_time = []

        for epoch in range(self.num_epochs): 

            
            # time measure            
            iter_times = []
            iter_st = None
            for i, (stream_idxs, stream_inputs, stream_targets) in enumerate(self.cl_dataloader):
                
                iter_en = time.perf_counter()
                if i > 0 and iter_st is not None:
                    iter_time = iter_en - iter_st
                    iter_times.append(iter_time)
                    #print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")

                    if i % 20 == 0:
                        print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")
                iter_st = time.perf_counter()
                

                stream_inputs = stream_inputs.to(self.device)
                stream_targets = stream_targets.to(self.device)
                stream_outputs = self.model(stream_inputs)
                loss = self.criterion(stream_outputs, stream_targets)
                
                if self.replay_dataset.is_filled() :
                    replay_idxs, replay_inputs, replay_targets, replay_logits, replay_ids = next(self.iter_r)

                    replay_inputs = replay_inputs.to(self.device)
                    replay_targets = replay_targets.to(self.device)
                    
                    replay_logits = replay_logits.to(self.device)
                    replay_outputs = self.model(replay_inputs)

                    if self.swap == True:
                        #print("BEFORE SWAP : ", replay_targets)
                        swap_idx, swap_targets, swap_ids = self.swap_manager.swap_determine(replay_idxs, replay_outputs, replay_targets, replay_ids)
                        
                        #print("AFTER SWAP : ", swap_targets)
                        self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist(), swap_ids.tolist())
                    
                    #
                    # need to match output dimension
                    #

                    if self.alpha_set > -1:
                        alpha = self.alpha_set
                    else:
                        alpha = self.alpha

                    loss += alpha * F.mse_loss(replay_outputs, replay_logits)

                    replay_idxs, replay_inputs, replay_targets, _, replay_ids = next(self.iter_r)
                    replay_inputs = replay_inputs.to(self.device)
                    replay_targets = replay_targets.to(self.device)
                    replay_outputs = self.model(replay_inputs)

                    if self.swap == True:
                        swap_idx, swap_targets, swap_ids = self.swap_manager.swap_determine(replay_idxs, replay_outputs, replay_targets, replay_ids)
                        self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist(), swap_ids.tolist())

                    loss += (1-alpha) * self.criterion(replay_outputs, replay_targets)
                    
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    
                #save files.
                    
                if self.swap == True and epoch == 0:
                    stream_data_batch = [self.stream_dataset.data[idx] for idx in stream_idxs]
                    self.swap_manager.saver.save(stream_data_batch, stream_targets.tolist(), stream_outputs.tolist())
           
                #update
                for idx, new_label, new_logit in zip(stream_idxs, stream_targets, stream_outputs):
                    new_data = self.stream_dataset.data[idx.item()]
                    sample_update_to_RB(self.replay_dataset, self.data_manager, new_data, new_label.item(), new_logit.data)
                    self.data_manager.increase_observed_samples()
                
            if epoch > 0:
                
                print(iter_times)
                self.curr_task_iter_time.append(np.mean(np.array(iter_times)))

            
            if epoch %5 ==0:
                print("\n---- TIME PROFILE ----")
                print("Swap : ", self.swap)
                if self.swap == True:
                    print("Swap workers : ", self.swap_num_workers)
                print("Fetch workers : ", self.cl_dataloader.num_workers)
                print(f"Iter time for epoch {epoch} / task {self.tasks_so_far} : ", self.curr_task_iter_time )
                print(f"Iter avg time for epoch {epoch} / task {self.tasks_so_far} : ", np.mean(np.array(self.curr_task_iter_time)) )
                print("----------------------\n")
            

            #if self.lr_scheduler is not None:
            #    self.lr_scheduler.step()

            #if epoch % 10 == 0 and epoch !=0:
            #    self.eval()
            
            print("lr {}".format(self.opt.param_groups[0]['lr']))
            
            self.loss_item.append(loss.item())

            if self.swap == True:
                print("epoch {}, loss {}, num_swap {}".format(epoch, loss.item(), self.swap_manager.get_num_swap()))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()
            else:
                print("epoch {}, loss {}".format(epoch, loss.item()))
            #print("time {}".format(en-st))
    
    def eval(self, get_entropy=False):
        
        self.model.eval()
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        
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

        for setp, ( imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            
            
            #get entropy of testset
            if self.swap==True and get_entropy == True:
                r, w = self.get_entropy(outputs, labels)
                r_entropy_test.extend(r)
                w_entropy_test.extend(w)
                
                logits_list.append(outputs)
                labels_list.append(labels)

            
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
        
        print("\n\ndata_manager claases_per_task : ", self.data_manager.classes_per_task)

        for task_id, task_classes in self.data_manager.classes_per_task.items():
            task_acc = np.mean(np.array(list(map(lambda x : class_accuracy[x] ,task_classes))))
            task_accuracy[task_id] = task_acc

        total_accuracy = 100 * correct / total
        self.model.train()

        
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
            

        return total_accuracy, task_accuracy, class_accuracy