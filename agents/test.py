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


class Test(Base):
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
        self.soft_incremental_acc = list()
        self.num_swap = list()

        self.criterion = torch.nn.CrossEntropyLoss()

    def before_train(self, task_id):
        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)

        self.cl_dataloader.update_loader()

        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        print("classes_so_far : ", self.classes_so_far)

        self.tasks_so_far += 1
        print("tasks_so_far : ", self.tasks_so_far)

        #self.model.Incremental_learning(self.classes_so_far)
        self.model.train()
        self.model.to(self.device)

        print("length of replay dataset : ", len(self.replay_dataset))

        if self.swap is True:
            self.swap_manager.before_train()

    def after_train(self):
        self.model.eval()

        if self.swap == True:
            self.swap_manager.after_train()
            for new_data, new_label in zip(self.stream_dataset.data, self.stream_dataset.targets):
                self.replay_dataset.store_image(new_data, new_label)

        multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset, self.model, 
                                        self.device, self.classes_so_far, self.base_transform)
        
        f = open('loss/'+self.filename+'.txt', 'a')
        f.write(str(self.loss_item)+"\n")
        f.close()

        f = open('accuracy/'+self.filename+'.txt', 'a')

        #temp acc
        print("SOFTMAX")
        curr_accuracy, task_accuracy, class_accuracy = self.eval(1)    
        print("soft_class_accuracy : ", class_accuracy)
        print("soft_task_accuracy : ", task_accuracy)
        print("soft_current_accuracy : ", curr_accuracy.item())
        
        self.soft_incremental_acc.append(curr_accuracy.item())
        print("incremental_accuracy : ", self.soft_incremental_acc)

        f.write("soft_class_accuracy : "+str(class_accuracy)+"\n")
        f.write("soft_task_accuracy : "+str(task_accuracy)+"\n")
        f.write("soft_incremental_accuracy : "+str(self.soft_incremental_acc)+"\n")
        f.close()

        
        f = open('num_swap/'+self.filename+'.txt', 'a')
        f.write(str(self.num_swap)+"\n")
        f.close()

        
        self.stream_dataset.clean_stream_dataset()

    def train(self):
        
        self.model.train()
        #self.reset_opt(self.tasks_so_far)
        self.opt = optim.SGD(self.model.parameters(), momentum=0.9, lr=2.0)
    
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                self.opt, [49,63], gamma=0.2
                            )

        for epoch in range(self.num_epochs): 

            for i, (idxs, inputs, targets) in enumerate(self.cl_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                
                #BCE instead of CE
                loss_value = self.criterion(outputs, targets)

                self.opt.zero_grad()
                loss_value.backward()
                self.opt.step()

            self.lr_scheduler.step()
            
            if epoch % 10 == 0 and epoch !=0:
                epoch_accuracy = self.eval(1)
            
            print("lr {}".format(self.opt.param_groups[0]['lr']))
            self.loss_item.append(loss_value.item())

            if self.swap == True:
                print("epoch {}, loss {}, num_swap {}".format(epoch, loss_value.item(), self.swap_manager.get_num_swap()))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()
            else:
                print("epoch {}, loss {}".format(epoch, loss_value.item()))
                
    
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
        task_accuracy = dict()

        for setp, ( imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
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
        self.model.train()
        return total_accuracy, task_accuracy, class_accuracy
    
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
