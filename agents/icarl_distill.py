from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import seaborn as sns

import numpy as np
import copy

import os
from agents.base import Base
from utils.sampling import multi_task_sample_update_to_RB
from lib import utils
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import time
import csv
from sklearn.manifold import TSNE

EPSILON = 1e-8



class make_dataset(Dataset):
    def __init__(self, data, targets, transform, ac_idx=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.actual_idx = ac_idx
        self.classes_in_dataset = set(targets)

    def __len__(self):
        assert len(self.data) == len(self.targets)
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        transformed_img = self.transform(img)
        label = self.targets[idx]

        return transformed_img, label

class ICarl_distill(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)

        if self.test_set == "cifar100":        
            self.base_transform = transforms.Compose([transforms.ToTensor(),
                                                    self.nomalize,])
            self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                        transforms.ToTensor(),
                                                        self.nomalize,])
        
        if self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            self.base_transform = transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                            ])
            self.base_transform = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    self.nomalize,])
            self.classify_transform=transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.RandomHorizontalFlip(p=1.),
                                                        transforms.ToTensor(),
                                                        self.nomalize,])
        
        self.old_model = None
        self.classes_so_far = 0
        self.tasks_so_far = 0
        self.nmc_incremental_acc = list()
        self.nmc_incremental_top5_acc = list()

        self.soft_incremental_acc = list()
        self.num_swap = list()

        self._herding_indexes = list()

        
        if 'distill' in kwargs:
            self.distill = kwargs['distill']
        else:
            self.distill = True
        print("DISTILL : ", self.distill)

        
        
        if 'alpha' in kwargs:
            self.alpha_set = float(kwargs['alpha'])
        else:
            # incremental alpha
            self.alpha_set = -1
        
        print("=================== Alpha is.....", self.alpha_set)

        
    def before_train(self, task_id):
        self.curr_task_iter_time = []
        self.model.eval()        
        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)

        self.cl_dataloader.update_loader()

        
        self.alpha = self.classes_so_far / (self.classes_so_far + len(self.stream_dataset.classes_in_dataset))

        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        self.model.Incremental_learning(self.classes_so_far)
        print("classes_so_far : ", self.classes_so_far)

        self.tasks_so_far += 1
        print("tasks_so_far : ", self.tasks_so_far)
        
        self.model.train()
        self.model.to(self.device)

        if self.old_model is not None:
            print("old model is available!")
            self.old_model.eval()
            self.old_model.to(self.device)

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
        
        if self.replay_dataset.sampling == "ringbuffer":
            multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset)

        else:
            self._herding_indexes, self._class_means = self.build_examplars(
                self._herding_indexes
            )
        
        
        #t-sne for icarl
        # self.get_tsne()

        print("NMC")
        avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc = self.eval_task(get_entropy=self.get_test_entropy)
        
        if self.swap == True:
            self.swap_manager.after_train(get_entropy=True)

        print("task_accuracy : ", task_top1_acc)
        if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
            print("task_top5_accuracy : ", task_top5_acc)

        print("current_accuracy : ", avg_top1_acc)
        if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
            print("current_top5_accuracy : ", avg_top5_acc)

        self.nmc_incremental_acc.append(avg_top1_acc)
        self.nmc_incremental_top5_acc.append(avg_top5_acc)

        print("incremental_top1_accuracy : ", self.nmc_incremental_acc)
        if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
            print("incremental_top5_accuracy : ", self.nmc_incremental_top5_acc)


        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        #f.write("class_accuracy : "+str(class_accuracy)+"\n")
        f.write("task_accuracy : "+str(task_top1_acc)+"\n")
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            f.write("task_top5_accuracy : "+str(task_top5_acc)+"\n")

        f.write("incremental_accuracy : "+str(self.nmc_incremental_acc)+"\n")
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            f.write("incremental_top5_accuracy : "+str(self.nmc_incremental_top5_acc)+"\n")
        f.close()

        
        
        f = open(self.result_save_path + self.filename + '_loss.txt', 'a')
        f.write(str(self.loss_item)+"\n")
        self.loss_item = []
        f.close()


        #f = open(self.result_save_path + self.filename + '_time.txt','a')
        #self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        #f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        #f.close()
        
        if self.swap==True:
            f = open(self.result_save_path + self.filename + f'_num_swap_{self.swap_base}.csv','a',newline="")
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.num_swap)
            f.close()
            self.num_swap = []
        

        self.old_model=copy.deepcopy(self.model)
        
        self.stream_dataset.clean_stream_dataset()

    
    def compute_examplar_mean(self, feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean

    def icarl_selection(self, features, nb_examplars):
        D = features.T
        D = D / (np.linalg.norm(D, axis=0) + 1e-8)
        mu = np.mean(D, axis=1)
        herding_matrix = np.zeros((features.shape[0],))

        w_t = mu
        iter_herding, iter_herding_eff = 0, 0

        while not (
            np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
        ) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if herding_matrix[ind_max] == 0:
                herding_matrix[ind_max] = 1 + iter_herding
                iter_herding += 1

            w_t = w_t + mu - D[:, ind_max]

        herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

        return herding_matrix.argsort()[:nb_examplars]
        
    def build_examplars(
        self, herding_indexes, memory_per_class=None, data_source="train"
    ):
        memory_per_class = self.replay_dataset.rb_size // self.classes_so_far
        print("MEM PER CLS : ", memory_per_class)
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self.classes_so_far, self.model.features_dim))

        for class_idx in range(self.classes_so_far):
            
            if self.tasks_so_far > 1 and class_idx not in self.stream_dataset.classes_in_dataset:
                inputs, targets, _ = self.replay_dataset.get_sub_data(class_idx)
                sub_dataset = make_dataset(inputs, targets, self.base_transform)
                loader = DataLoader(sub_dataset, batch_size=128, shuffle=False)

                sub_filped_dataset = make_dataset(inputs, targets, self.classify_transform)
                filped_loader = DataLoader(sub_filped_dataset, batch_size=128, shuffle=False)

                features, targets = utils.extract_features(self.model, loader, self.device)
                features_flipped, _ = utils.extract_features(
                    self.model, filped_loader, self.device
                )

            else: ##stream안에 있는 현재 클래스 일경우
                
                inputs, targets, _ = self.stream_dataset.get_sub_data(class_idx)
                sub_dataset = make_dataset(inputs, targets, self.base_transform)
                loader = DataLoader(sub_dataset, batch_size=128, shuffle=False)

                sub_filped_dataset = make_dataset(inputs, targets, self.classify_transform)
                filped_loader = DataLoader(sub_filped_dataset, batch_size=128, shuffle=False)

                features, targets = utils.extract_features(self.model, loader, self.device)
                features_flipped, _ = utils.extract_features(
                    self.model, filped_loader, self.device
                )

                selected_indexes = self.icarl_selection(features, memory_per_class)
                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            print(f"HERDING..{class_idx}")
            #print(herding_indexes)
            selected_indexes = herding_indexes[class_idx][:memory_per_class]
            herding_indexes[class_idx] = selected_indexes

            #print("feature shape : ",features.shape)
            #print("selected_indexes : ", selected_indexes)

            if self.tasks_so_far > 1 and class_idx not in self.stream_dataset.classes_in_dataset:
                selected_indexes = np.arange(0,memory_per_class)

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            for idx in selected_indexes:
                data_memory.append(inputs[idx])
                targets_memory.append(targets[idx])

            class_means[class_idx, :] = examplar_mean

        
        if self.swap == True:
            self.replay_dataset.data = self.manager.list(data_memory)
            self.replay_dataset.targets = self.manager.list(targets_memory)
        else:
            self.replay_dataset.data = data_memory
            self.replay_dataset.targets = targets_memory
        

        print(self.replay_dataset.targets)
        print("replay dataset len : ", len(self.replay_dataset))

        return herding_indexes, class_means

    #@profile
    def compute_loss(self, outputs, targets, imgs):
        org_target = targets.clone()
        targets = self.to_onehot(targets, self.classes_so_far).to(self.device)

        if self.distill == False or self.old_model is None:
            return F.binary_cross_entropy_with_logits(outputs, targets)
        else:
            #print("DISTILL ON")
            with torch.no_grad():
                old_targets=torch.sigmoid(self.old_model(imgs))
            
            targets_for_old = targets.clone()
            old_task_size = old_targets.shape[1]

            targets_for_old[..., :old_task_size] = old_targets
            loss1 = F.binary_cross_entropy_with_logits(outputs, targets_for_old)
            loss2 = F.binary_cross_entropy_with_logits(outputs, targets)        #ground truth loss
            
            if self.alpha_set > -1:
                alpha = self.alpha_set
            else:
                alpha = self.alpha
            print(f"Alpha is ... {alpha}")
            loss = (alpha) * loss1 + (1-alpha) * loss2
            #loss = F.binary_cross_entropy_with_logits(outputs, targets_for_old)
            return loss

    #@profile
    def train(self):

        self.model.train()
        self.reset_opt(self.tasks_so_far)
        
          #self.opt = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.00001)
        #self.lr_scheduler = None
        for epoch in range(self.num_epochs):
            

            # time measure            
            iter_times = []
            iter_st = None
            for i, (idxs, inputs, targets) in enumerate(self.cl_dataloader):
                iter_en = time.perf_counter()
                if i > 0 and iter_st is not None:
                    iter_time = iter_en - iter_st
                    iter_times.append(iter_time)
                    print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")
                    if i % 20 == 0:
                        print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")
                iter_st = time.perf_counter()

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                
                if self.swap == True and self.tasks_so_far > 1:
                    swap_idx, swap_targets = self.swap_manager.swap_determine(idxs,outputs,targets)
                    self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist())

                if self.distill == False:
                    targets = self.to_onehot(targets, self.classes_so_far).to(self.device)
                    loss_value = F.binary_cross_entropy_with_logits(outputs, targets)

                else:
                    loss_value = self.compute_loss(outputs, targets, inputs)
                
                self.opt.zero_grad()
                loss_value.backward()
                self.opt.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch > 0:
                
                print(iter_times)
                self.curr_task_iter_time.append(np.mean(np.array(iter_times)))

                print(f"Iter time for epoch {epoch} / task {self.tasks_so_far} : ", self.curr_task_iter_time )
                print(f"Iter avg time for epoch {epoch} / task {self.tasks_so_far} : ", np.mean(np.array(self.curr_task_iter_time)) )
                

            #print("lr {}".format(self.opt.param_groups[0]['lr']))
            self.loss_item.append(loss_value.item())
            
            if self.swap == True:
                print("epoch {}, loss {}, num_swap {}".format(epoch, loss_value.item(), self.swap_manager.get_num_swap()))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()
            else:
                print("epoch {}, loss {}".format(epoch, loss_value.item()))
    

    def get_tsne(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=True)

        features, targets_ = utils.extract_features(self.model, test_dataloader, self.device)
        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T


        tsne = TSNE(n_iter=2000, perplexity=50)
        tsne_results = tsne.fit_transform(features)

        print(tsne_results.shape)

        # Create the figure
        fig = plt.figure( figsize=(8,8) )
        plt.axis('off')
        sns.set_style('darkgrid')
        sns.scatterplot(tsne_results[:,0], tsne_results[:,1], hue=targets_, legend='full', palette=sns.color_palette("Spectral", as_cmap=True))
        ncol = int(self.classes_so_far / 10)
        lg = plt.legend(bbox_to_anchor=(1.05, 1.0), ncol=ncol, loc='upper left')
        plt.savefig(f'tsne/{self.filename}_tsne_task{self.tasks_so_far}.png', bbox_extra_artists=(lg,), 
            bbox_inches='tight')

    def eval_task(self, softmax=False, get_entropy=False):
        self.model.eval()
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        
        """
        if self.distill == False or self.swap == True:
            print("SOFTMAX EVAL!!")
            ypreds, ytrue = self.compute_accuracy_softmax(test_dataloader)
        else:
            ypreds, ytrue = self.compute_accuracy(self.model, test_dataloader, self._class_means)
        """
        if softmax==False:
            ypreds, ytrue = self.compute_accuracy(self.model, test_dataloader, self._class_means, get_entropy)
        else:
            ypreds, ytrue = self.compute_accuracy_softmax(test_dataloader)

        avg_top1_acc, task_top1_acc = self.accuracy_per_task(ypreds, ytrue, task_size=10, topk=1)
        avg_top5_acc, task_top5_acc = self.accuracy_per_task(ypreds, ytrue, task_size=10, topk=5)

        return avg_top1_acc, task_top1_acc, avg_top5_acc, task_top5_acc
    
    def compute_accuracy(self, model, loader, class_means, get_entropy=False):
        features, targets_ = utils.extract_features(model, loader, self.device)

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        # Compute score for iCaRL
        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        if self.swap==True and get_entropy == True:
            w_entropy_test = []
            r_entropy_test = []

            logits_list = []
            labels_list = []

        for setp, ( imgs, labels) in enumerate(loader):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)

            outputs = outputs.detach()
            
            #get entropy of testset
            if self.swap==True and get_entropy == True:
                print("GET TEST ENTROPY!!!!!!!!!!!!!!")
                r, w = self.get_entropy(outputs, labels)
                r_entropy_test.extend(r)
                w_entropy_test.extend(w)
                    
                logits_list.append(outputs)
                labels_list.append(labels)

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

            
        return score_icarl, targets_


    def compute_accuracy_softmax(self, loader):
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
        return round(correct_k / batch_size, 4)