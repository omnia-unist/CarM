from torch.nn import functional as F
import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import pandas as pd
from randaugment.randaugment import RandAugment
import numpy as np
import copy

import os
from agents.base import Base
from _utils.sampling import multi_task_sample_update_to_RB
from lib import utils
from scipy.spatial.distance import cdist
import time
from dataset.dataloader import RainbowStreamDataLoader, RainbowReplayDataLoader
from torch.utils.data import Dataset
import gc

class Rainbow(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)

        # transform
        self.test_transform = transform
        # data loader
        self.stream_loader = RainbowStreamDataLoader(self.stream_dataset, self.data_manager, self.cl_dataloader.num_workers, self.cl_dataloader.batch_size - (self.cl_dataloader.batch_size // 2) , swap)
        self.memory_loader = RainbowReplayDataLoader(self.replay_dataset, self.data_manager, self.cl_dataloader.num_workers, self.cl_dataloader.batch_size // 2, swap)

        # variables for evaluation
        self.classes_so_far = 0
        self.tasks_so_far = 0
        self.max_incremental_acc = list()
        self.incremental_acc = list()
        
        self.epoch_acc = list()
        self.max_epoch_acc = 0

        # memory sampling method to be used
        replay_dataset.sampling = "greedy_balance" # changed

        # specify the start and ending learning rates and weight decay to be used
        self.maxlr = 0.05
        self.minlr = 0.0005
        self.weight_decay = 1e-6

        # optimizer, scheduler, and criterion initialization
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.maxlr, momentum=0.9, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2, eta_min=self.minlr)

        # rainbow settings
        self.use_cutmix = True
        self.online = True
        self.use_only_memory = False
        self.uncert_metric = "vr_randaug"

    def before_train(self, task_id):
        self.curr_task_iter_time = []
        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        self.test_dataset.append_task_dataset(task_id)
        if self.swap is True:
            self.swap_manager.before_train()
            for new_label in self.stream_dataset.classes_in_dataset:
                sub_stream_data, sub_stream_label, sub_stream_idx = self.stream_dataset.get_sub_data(new_label)
                st = 0
                while True:
                    en = st + 20
                    self.swap_manager.saver.save(sub_stream_data[st:en], sub_stream_label[st:en])
                    st = en
                    if st > len(sub_stream_data):
                        break
                del sub_stream_data, sub_stream_label, sub_stream_idx
        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        print("classes_so_far : ", self.classes_so_far)

        # update the memory samples
        if(self.tasks_so_far == 0):
            multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset, self.model, 
                                        self.device, self.classes_so_far, self.transform)
        else:
            self.uncertainty_sampling(self.replay_dataset, self.stream_dataset, self.classes_so_far)
        # update the stream loader
        self.stream_loader.update_loader()
    
        self.tasks_so_far += 1
        print("tasks so far : ", self.tasks_so_far)

        self.model.Incremental_learning(self.classes_so_far)
        self.model.train()
        self.model.to(self.device)
        print("length of replay dataset : ", len(self.replay_dataset))
        self.replay_size = self.replay_dataset.rb_size

        self.stream_losses, self.replay_losses = list(), list()


    def after_train(self):
        self.model.eval()
        self.model.train()
        curr_accuracy, task_accuracy, class_accuracy = self.eval(get_entropy=self.get_test_entropy)

        print("max_class_accuracy : ", self.max_class_acc)
        print("max_task_accuracy : ", self.max_task_acc)
        # append the max epoch accuracy to list
        self.max_incremental_acc.append(self.max_epoch_acc)
        print("max_incremental_accuracy : ", self.max_incremental_acc)

        
        print("class_accuracy : ", class_accuracy)
        print("task_accuracy : ", task_accuracy)
        # append the max epoch accuracy to list
        self.incremental_acc.append(curr_accuracy)
        print("incremental_accuracy : ", curr_accuracy)


        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        
        f.write("max_class_accuracy : "+str(self.max_class_acc)+"\n")
        f.write("max_task_accuracy : "+str(self.max_task_acc)+"\n")
        f.write("max_incremental_accuracy : "+str(self.max_incremental_acc)+"\n")
        
        f.write("class_accuracy : "+str(class_accuracy)+"\n")
        f.write("task_accuracy : "+str(task_accuracy)+"\n")
        f.write("incremental_accuracy : "+str(curr_accuracy)+"\n")

        f.close()
        
        f = open(self.result_save_path + self.filename + '_time.txt','a')
        print("avg_iter_time : ", self.curr_task_iter_time)
        self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        f.close()

        
        if self.get_loss is True:
            f = open(self.result_save_path + self.filename + '_replay_loss.txt','a')
            f.write(str(self.replay_losses)+"\n")
            f.close()

            f = open(self.result_save_path + self.filename + '_stream_loss.txt','a')
            f.write(str(self.stream_losses)+"\n")
            f.close()
        


        # reset the epoch accuracies and other settings
        self.epoch_acc = list()
        self.max_epoch_acc = 0
        self.use_only_memory = False
        if self.online == False:
            self.memory_loader.update_loader()
            
        if self.swap == True:
            self.swap_manager.after_train()
        
        self.stream_dataset.clean_stream_dataset()
        gc.collect()
    
    def train(self):
        # for online setting 1st epoch with stream data and mem data, and rest using mem data
        for epoch in range(self.num_epochs):
            # Handle lr scheduling
            if epoch == 0: # Warm start of 1 epoch stream + memory
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.maxlr * 0.1
            elif epoch == 1: # Warm start of 1 epoch for memory
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.maxlr * 0.1
                # update the memory loader
                if self.online == True:
                    self.use_only_memory = True
                    self.memory_loader.update_loader()
            elif epoch == 2: # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.maxlr
            else: # Aand go!
                self.scheduler.step()
            

            # get the appropriate dataloader for the scenario
            if(self.tasks_so_far > 1 or self.use_only_memory == True):
                if(self.use_only_memory == False):
                    data_iterator = zip(self.stream_loader, self.cycle(self.memory_loader))
                    print("============using both memory and stream data===")
                else:
                    data_iterator = self.memory_loader
                    print("============using only memory data==============")
            else:
                data_iterator = self.stream_loader
                print("==========using only stream data=================")

            
            # time measure
            iter_times = []
            iter_st = None
            stream_loss, replay_loss = [],[]

            # training
            for i, (data) in enumerate(data_iterator):
                iter_en = time.perf_counter()
                if i > 0 and iter_st is not None:
                    iter_time = iter_en - iter_st
                    iter_times.append(iter_time)
                    #print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")

                    if i % 20 == 0:
                        print(f"EPOCH {epoch}, ITER {i}, TIME {iter_en-iter_st}...")
                iter_st = time.perf_counter()

                # get the memory and stream data
                if len(data) == 2:
                    stream_data, mem_data = data
                    _, input_stream, target_stream = stream_data
                    idxs, input_memory, target_memory = mem_data
                    inputs = torch.cat((input_stream, input_memory))
                    targets = torch.cat((target_stream, target_memory))
                else:
                    idxs, inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # whether to use cutmix or not
                if(self.use_cutmix):
                    inputs, labels_a, labels_b, lam = self.cutmix_data(x=inputs, y=targets, alpha=0.5)
                    outputs = self.model(inputs)
                    loss_value = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    outputs = self.model(inputs)
                    loss_value = self.criterion(outputs, targets)

                if self.get_loss == True:
                    loss_v = torch.nn.CrossEntropyLoss(reduction='none')(outputs,targets)
                    get_loss = loss_v.clone().detach()
                    #get_loss = loss_ext.view(loss_ext.size(0), -1)
                    #get_loss = loss_ext.mean(-1)
                    replay_idxs = (idxs < len(self.replay_dataset)).squeeze().nonzero(as_tuple=True)[0]
                    stream_idxs = (idxs >= len(self.replay_dataset)).squeeze().nonzero(as_tuple=True)[0]
                    #print(get_loss)
                    #print(stream_idxs)
                    stream_loss.append(get_loss[stream_idxs].mean(-1).item())
                    if get_loss[replay_idxs].size(0) > 0:
                        replay_loss.append(get_loss[replay_idxs].mean(-1).item())

                    
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
                
                if self.swap == True and epoch > 0:
                    #if self.dynamic == True and epoch < (len(self.stream_dataset) * (self.tasks_so_far-1) / self.replay_size) * (1/self.swap_manager.threshold) * 5: # dynamic_ver. at least five epochs for all dataset
                    if self.dynamic == True and epoch <= int(self.num_epochs * 0.5):
                        print("random")
                        swap_idx, swap_targets = self.swap_manager.random(idxs,outputs,targets)
                        self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist())

                    else:
                        swap_idx, swap_targets = self.swap_manager.swap_determine(idxs, outputs, targets)
                        self.swap_manager.swap(swap_idx.tolist(), swap_targets.tolist())

            # evaluate on every epoch...
            if self.swap == True:
                print("epoch {}, loss {}, num_swap {}".format(epoch, loss_value.item(), self.swap_manager.get_num_swap()))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()
            
            if epoch > 0:    
                #print(iter_times)
                self.curr_task_iter_time.append(np.mean(np.array(iter_times)))
            
            
            curr_accuracy, task_accuracy, class_accuracy = self.eval()
            print("class_accuracy : ", class_accuracy)
            print("task_accuracy : ", task_accuracy)
            print("current_epoch_accuracy : ", curr_accuracy.item())
            self.epoch_acc.append(curr_accuracy.item())
            
            self.stream_losses.append(np.mean(np.array(stream_loss)))
            self.replay_losses.append(np.mean(np.array(replay_loss)))
            
            print("incremental_epoch_accuracy : ", self.epoch_acc)
            if curr_accuracy >= self.max_epoch_acc:
                self.max_epoch_acc = curr_accuracy.item()
                self.max_class_acc = class_accuracy
                self.max_task_acc = task_accuracy


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
    

    def cutmix_data(self, x, y, alpha=1.0, cutmix_prob=0.5):
        assert(alpha > 0)
        # generate mixed sample
        lam = np.random.beta(alpha, alpha)

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        if torch.cuda.is_available():
            index = index.to(self.device)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    def cycle(self, iterable):
    # iterate with shuffling
        while True:
            for i in iterable:
                yield i
        
    def uncertainty_sampling(self, samples_mem, samples_stream, num_class):
        num_of_transforms = 4
        data_all = []
        targets_all = []
        uncertainity = []
        for i in range(num_of_transforms + 1):
          uncertainity.append([])
        # get a list of all samples in memory and stream
        for i in range(len(samples_mem)):
            data_all.append(samples_mem.data[i])
            targets_all.append(samples_mem.targets[i])
        for i in range(len(samples_stream)):
            data_all.append(samples_stream.data[i])
            targets_all.append(samples_stream.targets[i])
        for i in range(len(samples_mem) + len(samples_stream)):
            for j in range(num_of_transforms + 1):
                uncertainity[j].append(0)
        samples = {'image': data_all,
                    'label': targets_all,
                    'uncert_0': uncertainity[0], 'uncert_1': uncertainity[1], 'uncert_2': uncertainity[2],'uncert_3': uncertainity[3],
                    "uncertainty": uncertainity[4]}
        # get the uncertainity metric
        samples = self.montecarlo(samples, uncert_metric=self.uncert_metric)
        sample_df = pd.DataFrame(samples)
        mem_per_cls = len(self.replay_dataset) // num_class
        ret = []
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")
        # put the chosen images in the replay buffer
        for i in range(len(ret)):
            samples_mem.data[i] = ret[i]['image']
            samples_mem.targets[i] = ret[i]['label']

    def montecarlo(self, candidates, uncert_metric="vr"):
        transform_cands = []
        print(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "vr":
            transform_cands = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
            ]
        elif uncert_metric == "vr_randaug":
            for _ in range(4):
                transform_cands.append(RandAugment())
        n_transforms = len(transform_cands)
        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            candidates = self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")
        candidates = self.variance_ratio(candidates, n_transforms)
        return candidates

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 32
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            infer_df, transform=infer_transform
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )
        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()
                for i, cert_value in enumerate(logit):
                    infer_list[uncert_name][batch_size * n_batch + i] = 1 - cert_value
        return infer_list

    def variance_ratio(self, samples, cand_length):
        for idx in range(len(samples['image'])):
            vote_counter = torch.zeros(samples["uncert_0"][idx].size(0))
            for i in range(cand_length):
                top_class = int(torch.argmin(samples[f"uncert_{i}"][idx]))  # uncert argmin.
                vote_counter[top_class] += 1
            assert vote_counter.sum() == cand_length
            samples["uncertainty"][idx] = (1 - vote_counter.max() / cand_length).item()
        return samples

class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.data_frame['label'][idx]
        image = self.data_frame['image'][idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample
