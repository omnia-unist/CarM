from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn as nn
import csv
from agents.base import Base

class StreamER(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, 
                test_dataset, filename, **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, 
                        test_dataset, filename, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        
        self.trained_class_set = set()
        self.classes_so_far = 0
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00001)
        
        f = open(f'accuracy/{self.filename}.csv','a', newline='')
        wr = csv.writer(f)
        wr.writerow(['average accuracy'])
        f.close()


    def get_class_so_far(self, y):
        self.trained_class_set.update(y.tolist())
        return max(self.trained_class_set)
    
    def increase_fc_output(self, labels):
        num_classes_in_curr_batch = self.get_class_so_far(labels) + 1
        if self.classes_so_far < num_classes_in_curr_batch:
            self.classes_so_far = num_classes_in_curr_batch
            self.model.Incremental_learning(self.classes_so_far)
            self.model.train()
            self.model.to(self.device)

    #def after_train(self):
    #    self.eval()
    
    def train(self, i, task_id, x, y, x_, y_):
        """
        x : stream data
        y : stream label
        x_ : replay data
        y_ : replay label
        """

        self.model.train()
        
        x = x.to(self.device)
        y = y.to(self.device)

        if x_ is not None:
            x_ = x_.to(self.device)
            y_ = y_.to(self.device)
            images = torch.cat([x,x_])
            targets = torch.cat([y,y_])
        else:
            images = x
            targets = y

        self.increase_fc_output(targets)

        images = images.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(images)
        
        if self.swap == True and i>1:
            print(i, "swap queue is filled")
            swap_in_batch = self.swap_manager.swap_determine(outputs,targets)
            self.cl_dataloader.send_what_to_swap(swap_in_batch)
        
        loss = self.criterion(outputs, targets)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        print(loss.item())

        print("\n\n\n\n\nSTREAM TRAINING!!!!!!!!!\n\n\n\n\n")
        print("labels : ", targets)
        print("unique_labels : ", torch.unique(targets))
        self.eval(torch.unique(targets))


    def eval(self, labels_to_add):
        for label in labels_to_add:
            print("task_id.item : ",label.item())
            self.test_dataset.append_class_dataset(label.item())
        test_dataloader = DataLoader(self.test_dataset, batch_size = 10, shuffle=False)

        class_correct = list(0. for i in range(100))
        class_total = list(0. for i in range(100))
        per_acc = []

        self.model.eval()

        with torch.no_grad():
            for (images, labels) in test_dataloader:
                #print("test label", labels)
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                                    
                _, predicted = torch.max(outputs, 1)
                #print(predicted)

                c = (predicted == labels).squeeze()


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

            per_acc.append(class_acc)

        print("avg accuracy : ", np.average(np.array(per_acc)))
        

        f = open(f'accuracy/{self.filename}.csv','a', newline='')
        wr = csv.writer(f)
        wr.writerow([np.average(np.array(per_acc))])
        f.close()
