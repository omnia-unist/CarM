from torch.nn import functional as F
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import copy

import os
from agents.base import Base
from _utils.sampling import multi_task_sample_update_to_RB
from lib import utils

from scipy.spatial.distance import cdist
import time

import csv

class PureER(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)
        
        self.base_transform = transforms.Compose(
               [transforms.ToTensor(),
               transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                    transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classes_so_far = 0
        self.tasks_so_far = 0
        
        self.nmc_incremental_acc = list()
        self.soft_incremental_top1_acc = list()
        self.soft_incremental_top5_acc = list()
        self.num_swap = list()

        self.criterion = torch.nn.CrossEntropyLoss()

    def before_train(self, task_id):
        
        self.curr_task_iter_time = []
        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)

        self.cl_dataloader.update_loader()

        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        print("classes_so_far : ", self.classes_so_far)

        self.tasks_so_far += 1
        print("tasks_so_far : ", self.tasks_so_far)

        self.model.Incremental_learning(self.classes_so_far)
        self.model.train()
        self.model.to(self.device)

        print("length of replay dataset : ", len(self.replay_dataset))

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

    def after_train(self):
        self.model.eval()

        if self.swap == True:
            self.swap_manager.after_train()

        multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset)
        
        #temp acc
        print("SOFTMAX")
                
        avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc = self.eval_task()
        
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

        """
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
        """

        
        
        if self.swap==True:
            f = open(self.result_save_path + self.filename + f'_num_swap_{self.swap_base}.csv','a',newline="")
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.num_swap)
            f.close()
            self.num_swap = []


            f = open(self.result_save_path + self.filename + '_distribution.txt', 'a')
            f.write("incremental_accuracy : "+str(self.swap_manager.swap_class_dist)+"\n")
            f.close()
            self.swap_manager.reset_swap_class_dist()
            
        
        """nvidia
        f = open('num_swap/'+self.filename+'.txt', 'a')
        f.write(str(self.num_swap)+"\n")
        f.close()
        """
        
        self.stream_dataset.clean_stream_dataset()

    def train(self):
        
        self.model.train()
        self.reset_opt(self.tasks_so_far)
        for epoch in range(self.num_epochs): 

            
            # time measure            
            iter_times = []
            iter_st = None
            for i, (idxs, inputs, targets) in enumerate(self.cl_dataloader):
                iter_en = time.perf_counter()
                if i > 0 and iter_st is not None:
                    iter_time = iter_en - iter_st
                    print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")
                    iter_times.append(iter_time)
                    if i % 20 == 0:
                        print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")
                iter_st = time.perf_counter()


                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                
                if self.swap == True and self.tasks_so_far > 1:
                    swap_idx, swap_targets = self.swap_manager.swap_determine(idxs,outputs,targets)
                    
                    self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist())

                #BCE instead of CE
                targets = self.to_onehot(targets, self.classes_so_far).to(self.device)
                loss_value = F.binary_cross_entropy_with_logits(outputs, targets)
                
                self.opt.zero_grad()
                loss_value.backward()
                self.opt.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            #epoch_accuracy = self.eval(1)
            print("lr {}".format(self.opt.param_groups[0]['lr']))
            self.loss_item.append(loss_value.item())
            if epoch > 0:
                
                print(iter_times)
                self.curr_task_iter_time.append(np.mean(np.array(iter_times)))

            
            if self.swap == True:
                print("epoch {}, loss {}, num_swap {}".format(epoch, loss_value.item(), self.swap_manager.get_num_swap()))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()
            else:
                print("epoch {}, loss {}".format(epoch, loss_value.item()))
                
    
    
    def eval_task(self):
        self.model.eval()
        
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        ypreds, ytrue = self.compute_accuracy(test_dataloader)

        
        avg_top1_acc, task_top1_acc = self.accuracy_per_task(ypreds, ytrue, task_size=10, topk=1)
        avg_top5_acc, task_top5_acc = self.accuracy_per_task(ypreds, ytrue, task_size=10, topk=5)

        return avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc

    
    def compute_accuracy(self, loader):
        ypred, ytrue = [], []

        for setp, ( imgs, labels) in enumerate(loader):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)

            outputs = outputs.detach()

            ytrue.append(labels.numpy())
            ypred.append(torch.softmax(outputs, dim=1).cpu().numpy())

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
        return round(correct_k / batch_size, 3)

    """
    def eval(self, mode):
        
        self.model.eval()
        
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        
        if mode==0:
            print("compute NMS")
        self.model.eval()
        
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

        total_accuracy = 100 * correct / total
        total_top5_accuracy = np.mean(np.array(top5_accuracy))
        self.model.train()
        return total_accuracy, total_top5_accuracy, task_accuracy, class_accuracy
    
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
