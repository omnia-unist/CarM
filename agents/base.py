from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from torchvision import transforms
import multiprocessing as python_multiprocessing

import random
import networks
from networks.myNetwork import network
from networks.resnet_cbam import resnet18_cbam, resnet34_cbam
from networks.resnet_official import resnet18
from networks.pre_resnet import PreResNet
from networks.resnet_for_cifar import resnet32
from networks.der_resnet import resnet18 as der_resnet18
from networks.tiny_resnet import ResNet18 as tiny_resnet18
from networks.densenet import DenseNet as densenet
from lib.factory import get_optimizer
from lib.swap_manager import SwapManager
from lib.utils import _ECELoss

class Base(object):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):
        #self.model = model

        #set seed
        #random_seed = 0
        #torch.manual_seed(random_seed)
        #torch.cuda.manual_seed(random_seed)
        #np.random.seed(random_seed)
        #random.seed(random_seed)
        
        self.opt_name = opt_name
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay

        self.device = device
        self.num_epochs = num_epochs
        self.swap = swap
        self.transform = transform
        self.test_set = test_set
        self.ece_loss = _ECELoss().to(self.device)

        self.set_nomalize()
        
        self.loss_item = list()
        
        self.data_manager = data_manager
        self.stream_dataset = stream_dataset
        self.replay_dataset = replay_dataset
        self.cl_dataloader = cl_dataloader
        self.test_dataset = test_dataset
        self.filename = filename
 
        self.num_swap = list()
        self.avg_iter_time = list()


        if 'result_save_path' in kwargs:
            self.result_save_path = kwargs['result_save_path'].strip()
            if self.result_save_path[-1:] != "/":
                self.result_save_path += "/"
            print(self.result_save_path)
        else:
            self.result_save_path = './exp_results/test/'

            
            
        if model == "resnet18":
            #self.model = network(resnet18_cbam())
            self.model = network(resnet18())
        elif model == "resnet34":
            print("RESNET34!!!!")
            self.model = network(resnet34_cbam())
        elif model == "resnet32":
            #self.model = PreResNet(32)
            self.model = network(PreResNet(32))
        elif model == "der_resnet":
            
            if self.test_set == "imagenet1000":
                self.model = der_resnet18(1000)
            elif self.test_set == "tiny_imagenet":
                self.model = der_resnet18(200)
            elif self.test_set == "cifar100":
                self.model = der_resnet18(100)
            else:
                self.model = der_resnet18(10)
            
        elif model == "tiny_resnet":
            self.model = network(tiny_resnet18())
            #self.model = tiny_resnet18()
        #elif model == "aser_resnet":
        #    self.model = network(aser_resnet18(10))
        elif model == "densenet":
            self.model = network(densenet())


        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        else:
            self.seed = None
        
        if 'get_loss' in kwargs:
            self.get_loss = kwargs['get_loss']
        else:
            self.get_loss = False
            
        if 'get_train_entropy' in kwargs:
            self.get_train_entropy = kwargs['get_train_entropy']
        else:
            self.get_train_entropy = False
        
        if 'get_test_entropy' in kwargs:
            self.get_test_entropy = kwargs['get_test_entropy']
        else:
            self.get_test_entropy = False
        
        
        #elif model == "cifar_resnet":
        #    self.model = network(resnet_rebuffi(32))
        #elif model == "icarl_resnet":
        #    self.model = network(make_icarl_net())

        if self.swap == True:
            
            if 'dynamic' in kwargs:
                self.dynamic = kwargs['dynamic']
            else:
                self.dynamic = False
                    
            if 'swap_base' in kwargs:
                self.swap_base = kwargs['swap_base']
            else:
                self.swap_base = 'all'
                
            if 'threshold' in kwargs:
                threshold = float(kwargs['threshold'])
            else:
                threshold = 0.5

            
            if 'swap_workers' in kwargs:
                self.swap_num_workers = int(kwargs['swap_workers'])
            else:
                self.swap_num_workers = 1
                
            print(self.swap_base, threshold)
            
            """
            if self.test_set in ["imagenet", "imagenet1000"]:
                imagenet = True
            else:
                imagenet = False
            """
            #make save worker


            if 'store_ratio' in kwargs:
                self.store_ratio = float(kwargs['store_ratio'])

                """
                if self.test_set == "tiny_imagenet":
                    store_budget = int(100000 * self.store_ratio)
                elif self.test_set == "mini_imagenet":
                    store_budget = int(50000 * self.store_ratio)
                elif self.test_set == "imagenet100":
                    store_budget = int(129395 * self.store_ratio)
                elif self.test_set == "imagenet1000" or self.test_set == "imagenet":
                    store_budget = int(1281168 * self.store_ratio)
                elif self.test_set == "cifar100":
                    store_budget = int(50000 * self.store_ratio)
                elif self.test_set == "cifar10":
                    store_budget = int(50000 * self.store_ratio)
                else:
                    store_budget = None
                """
                store_budget = int(self.replay_dataset.rb_size * self.store_ratio)
                self.store_budget = store_budget
            else:
                self.store_budget = None


            print("=========================================STORE BUDGET : ", self.store_budget)

            """
            self.swap_manager = SwapManager(self.replay_dataset, self.swap_num_workers, 
                                            self.swap_base, threshold=threshold, store_budget=self.store_budget,
                                            swapping_set=self.swapping_set)
            """
        
        if (self.swap == True and self.swap_num_workers> 0) or self.cl_dataloader.num_workers > 0:
            self.manager = python_multiprocessing.Manager()
            
            print('MANAGER_1 PID:', self.manager._process.ident)
            self.replay_dataset.data = self.manager.list(self.replay_dataset.data)
            self.replay_dataset.targets = self.manager.list(self.replay_dataset.targets)
            
            
            self.replay_dataset.tracker = self.manager.list(self.replay_dataset.tracker)
            if hasattr(self.replay_dataset, 'logits'):
                self.replay_dataset.logits = self.manager.list(self.replay_dataset.logits)
        
        if self.swap == True:
            self.swap_manager = SwapManager(self.replay_dataset, self.swap_num_workers, 
                                            self.swap_base, threshold=threshold, store_budget=self.store_budget, 
                                            filename=self.filename, result_save_path = self.result_save_path, get_entropy=self.get_train_entropy, seed=self.seed)
            

    def to_onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
        return onehot
    
    def set_nomalize(self):
        if self.test_set in ["cifar10"]:
            self.nomalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2615])
        if self.test_set in ["cifar100"]:
            self.nomalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                std=[0.2009, 0.1984, 0.2023])
        elif self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            self.nomalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    def get_entropy(self, outputs, targets):
        
        if self.get_test_entropy == False:
            return

        print("GET TEST ENTROPY IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        softmax = torch.nn.Softmax(dim=1)
        soft_output = softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_entropy = entropy[r_predicted]

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_entropy = entropy[w_predicted]
        
        return r_entropy.tolist(), w_entropy.tolist()

    #hard coded
    def reset_opt(self, step=None):
        if self.test_set == "cifar100":
            #icarl setting
            self.opt = torch.optim.SGD(self.model.parameters(), lr=2.0, momentum=0.9, weight_decay=0.00001)
            
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    self.opt, [49,63], gamma=0.2
                                )
            if self.num_epochs > 80:
                lr_change_point = list(range(0,self.num_epochs,40))
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, lr_change_point, gamma=0.25)
            
        elif self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            #icarl setting
            print("ICARL IMAGENET OPT SET...")
            self.opt = torch.optim.SGD(self.model.parameters(), lr=2.0, momentum=0.9,  weight_decay=0.00001) #imagenet
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [20,30,40,50], gamma=0.20) #imagenet
        
        else:
            self.opt = torch.optim.SGD(self.model.parameters(), lr=0.10, momentum=0.9, weight_decay=0.0001)
            self.lr_scheduler = None
    

    def before_train(self):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError
    def after_train(self):
        raise NotImplementedError
