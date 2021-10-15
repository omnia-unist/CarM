from numpy.core.numeric import full
import torch
import multiprocessing as python_multiprocessing
import asyncio
import os
import random
import pickle

from PIL import Image
import itertools
import numpy as np

import threading
import queue
from array import array 
import io
import math

from lib.save import DataSaver
from line_profiler import LineProfiler

import directio, io, os

#torch.multiprocessing.set_sharing_strategy('file_system')

#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (2046, rlimit[1]))

class _DatasetSwapper(object):
    def __init__(self, dataset, saver):
        if hasattr(dataset, 'replay_dataset'):
            self.dataset = dataset.replay_dataset
        else:
            self.dataset = dataset
        self.saver = saver

    """
    async def _get_logit(self, logit_filename):
        with open(logit_filename, 'rb') as f:
            return pickle.load(f)

    async def _get_img(self, filename):
        with open(filename, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img
    """

    async def _get_logit(self, logit_filename):
        f = os.open( logit_filename, os.O_RDONLY | os.O_DIRECT)
        os.lseek(f,0,0)
        actual_size = os.path.getsize(logit_filename)
        block_size = 512 * math.ceil(actual_size / 512)
        fr = directio.read(f, block_size)
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        
        logit = pickle.load(data)

        return logit

    async def _get_img(self, filename):
        f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

        os.lseek(f,0,0)
        actual_size = os.path.getsize(filename)
        block_size = 512 * math.ceil(actual_size / 512)
        fr = directio.read(f, block_size)
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        

        img = Image.open(data)
        img = img.convert('RGB')
        
        #print("FILE OPEN TIME : ", en-st)

        return img
    
    
    async def _get_file_list(self, path):
        list_dir = os.listdir(path)
        return list_dir

    async def _get_data(self, idx, filename, data_id=None):
        if hasattr(self.dataset, 'logits'):
            if 'png' in filename:
                logit_filename = filename.split('.png')[0] + '.pkl'
            elif 'pkl' in filename:
                logit_filename = filename
                filename = filename.split('.pkl')[0] + '.png'

            try:
                logit = await self._get_logit(logit_filename)
            except Exception as e:
                #print(self.saver.num_file_for_label)
                print(e)
                
                return False
        
        try:           
            img = await self._get_img(filename)
            
            if hasattr(self.dataset, 'logits') and data_id is not None:
                if self.dataset.tracker[idx] != data_id:
                    #print("ID NOT MATCH")
                    return False
            elif data_id is not None:
                idx = (self.dataset.tracker).index(data_id)
            
            self.dataset.data[idx] = img
            if hasattr(self.dataset, 'logits'):
                self.dataset.logits[idx] = logit
            
            #print("SWAPPED")
            return True

        except Exception as e:
            print(e)
            return False

    async def _swap_main(self, label, swap_idx, data_id=None):
        
        path_curr_label = self.dataset.rb_path + '/' + str(label)
        """
        try:
            replace_file_list = await self._get_file_list(path_curr_label)
        except Exception as e:
            print(e)
            return False
        
        replace_file = path_curr_label + '/' + random.choice(replace_file_list)
        """
        try:
            #print("swap w num_file_for_label : ", self.saver.num_file_for_label)
            #print("\n")

            replace_file = path_curr_label + '/' + str(random.randint(1,self.saver.num_file_for_label[label])) +'.png'
        except Exception as e:
            #print("RANDINT")
            print(e)
            return False

        return await self._get_data(swap_idx, replace_file, data_id)
    
    async def _swap(self, what_to_swap, labels, data_ids=None):
        
        if data_ids is not None:
            cos = [ self._swap_main(label, idx, data_id) for label, idx, data_id in zip(labels, what_to_swap,data_ids) ]        
        else:
            cos = [ self._swap_main(label, idx) for label, idx in zip(labels, what_to_swap) ]
        res = await asyncio.gather(*cos)
        #print(res)

        #print(f"LEN : {len(what_to_swap)}, SWAP TIME : {en-st}")

        return res

        #print("\n")
import time
def wrap(worker_id, dataset, swap_queue, done_event):
    
    prof = LineProfiler()
    prof.add_function(_DatasetSwapper._get_data)
    prof.add_function(_DatasetSwapper._get_file_list)
    prof.add_function(_DatasetSwapper._swap_main)
    prof.runcall(adaptive_swap_loop, worker_id, dataset, swap_queue, done_event)
    prof.print_stats()
    prof.dump_stats('xxx%d.prof'%worker_id)


def adaptive_swap_loop(worker_id, dataset, saver, swap_queue, done_event, seed=None):
    #torch.set_num_threads(1)
    #random.seed(seed)
    #torch.manual_seed(seed)

    if seed is not None:
        print("============== SEED SET IN SWAP WORKERS!!!!!!!!!!!! =======================")
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    swapper = _DatasetSwapper(dataset,saver)
    
    print("\n\n===========in swap worker==============\n")
    print("SWAP WORKER PID : ", os.getpid())
    print("=====================================\n\n")
    
    while True:
        try:
            swap_idx, labels, data_ids = swap_queue.get(timeout=5.0)
            
        except queue.Empty:
            continue
        
        #print(swap_idx, labels, data_ids)
        if swap_idx is None and labels is None:
            print("Thread loop break!!!")
            assert done_event.is_set()
            break
        
        elif done_event.is_set():
            continue
        
        #swap_queue.task_done()

        #swapper._swap(swap_idx, labels, data_ids)

        # 여기서 set에 있는 swap_idx넣기
        #swapping_set.extend(swap_idx)
        #print("before : ", swapping_set)

        if data_ids is not None:
            swap_res = asyncio.run(swapper._swap(swap_idx, labels, data_ids))
        else:
            swap_res = asyncio.run(swapper._swap(swap_idx, labels))
        
        # 여기서 set에 있는 swap_idx없애기
        #for s_idx in swap_idx:
        #    swapping_set.remove(s_idx)
        #print("after : ", swapping_set)

        del swap_idx, labels, data_ids

class SwapManager(object):
    total_count = 0
    overlap_count = 0
    manager = python_multiprocessing.Manager()
    #
    # swapping_set = manager.list()

    def __init__(self, replay_dataset, num_workers, swap_base, store_budget=None, get_loss=False, get_entropy=False, seed=None, **kwargs):
        print(num_workers, swap_base, kwargs)
        self.swap_base = swap_base
        self.swap_determine = self.swap_policy(swap_base)
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_workers = num_workers
        self.dataset = replay_dataset
        self.num_swap = 0
        self.saver = DataSaver(replay_dataset.rb_path, store_budget, seed)
        self.agent = self.dataset.agent
        self._swap_loss = None
        self.new_classes = None
        self.seed = seed

        self.swap_class_dist = {}

        if num_workers == 0:
            self._swapper = _DatasetSwapper(self.dataset, self.saver)

        if 'threshold' in kwargs:
            self.threshold = float(kwargs['threshold'])
        
        if 'filename' in kwargs:
            self.filename = kwargs['filename']

        if 'result_save_path' in kwargs:
            self.result_save_path = kwargs['result_save_path']
        
        self._get_loss = get_loss
        self._get_entropy = get_entropy

        self.data_correct_entropy = []
        self.data_wrong_entropy = []

        self.data_correct_loss = []
        self.data_wrong_loss = []
        
        self.stream_loss = []
        self.replay_loss = []

    """
    @classmethod
    def is_overlapped(cls, replay_idx):
        #print(cls.swapping_set)

        for re_idx in replay_idx:
            if re_idx in cls.swapping_set:
                cls.overlap_count += 1
            cls.total_count += 1
    """
        
    def before_train(self):  

        if self.agent in ["der","derpp", "tiny","aser"]:
            self.rp_len = None
        else:
            self.rp_len = len(self.dataset)

        print("\n===================")
        print("REPLAY BUFFER SIZE (profiled by swap worker): ", self.rp_len)
        print("===================\n")

        self.saver.before_train()
        
        if self.num_workers > 0 :
            #swap process
            self._swap_workers=[]
            self._swap_queues=[]
            self._swap_done_event = python_multiprocessing.Event()
            self._swap_worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))

            for i in range(self.num_workers):
                swap_queue = python_multiprocessing.Queue()        
                swap_worker = python_multiprocessing.Process(
                    target = adaptive_swap_loop,
                    args=(i, self.dataset, self.saver, swap_queue, self._swap_done_event, self.seed)
                )
                swap_worker.daemon = True
                swap_worker.start()
                
                self._swap_workers.append(swap_worker)
                self._swap_queues.append(swap_queue)
        """
        if self.num_workers > 0 :

            #swap thread
            self._swap_workers=[]
            self._swap_queues=[]
            self._swap_done_event = threading.Event()
            self._swap_worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))

            for i in range(self.num_workers):
                swap_queue = queue.Queue()        
                swap_worker = threading.Thread(
                    target = adaptive_swap_loop,
                    args=(i, self.dataset, swap_queue, self._swap_done_event)
                )
                swap_worker.daemon = True
                swap_worker.start()
                
                self._swap_workers.append(swap_worker)
                self._swap_queues.append(swap_queue)
        """
    def after_train(self, get_loss=False, get_entropy=False):
        #shudown swap process
        if self.num_workers > 0:
            #for sq in self._swap_queues:
            #    print("joining...")
            #    sq.join()
            
            self._swap_done_event.set()
            for sq in self._swap_queues:
                sq.put((None, None, None))
            
            for s in self._swap_workers:
                s.join() #timeout?

            for sq in self._swap_queues:
                sq.cancel_join_thread()
                sq.close()

            
            print("SWAP SHUTDOWN")

            self.saver.after_train()
        
        if get_loss == True or self._get_loss == True:
            print("RECORD LOSS!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            f = open(self.result_save_path + self.filename + '_correct_train_loss.txt', 'a')
            f.write(str(self.data_correct_loss)+"\n")
            f.close()
            
            f = open(self.result_save_path + self.filename +'_wrong_train_loss.txt', 'a')
            f.write(str(self.data_wrong_loss)+"\n")
            f.close()
    
        self.data_correct_loss = []
        self.data_wrong_loss = []

        if get_entropy == True or self._get_entropy == True:
            print("RECORD ENTROPY!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            f = open(self.result_save_path + self.filename + '_correct_train_entropy.txt', 'a')
            f.write(str(self.data_correct_entropy)+"\n")
            f.close()
            
            f = open(self.result_save_path + self.filename + '_wrong_train_entropy.txt', 'a')
            f.write(str(self.data_wrong_entropy)+"\n")
            f.close()
    
        self.data_correct_entropy = []
        self.data_wrong_entropy = []
    

        """
        if self.agent in ["bic","er","bic_distill", "icarl_distill"]:
            print("RECORD ENTROPY!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            f = open(f'entropy/{self.agent}_{self.filename}_correct_entropy_softmax_swapped.txt', 'a')
            f.write("correct entropy : "+ "\n" + str(self.data_correct_entropy)+"\n")
            f.close()
            
            f = open(f'entropy/{self.agent}_{self.filename}_wrong_entropy_softmax_swapped.txt', 'a')
            f.write("wrong entropy : "+ "\n" + str(self.data_wrong_entropy)+"\n")
            f.close()
        
            self.data_correct_entropy = []
            self.data_wrong_entropy = []
        
        print("==================================\n")
        print("HOW MANY ARE OVERLAPPED FOR SWAP\n")
        print("overlapped count : ", self.overlap_count)
        print("total count : ", self.total_count)
        print("ratio : ", self.overlap_count / self.total_count)
        print("==================================\n")
        """

    def get_num_swap(self):
        return self.num_swap

    def reset_num_swap(self):
        self.num_swap = 0

    
    def reset_swap_class_dist(self):
        self.swap_class_dist = {}

    #@profile
    def swap(self, what_to_swap, labels, data_ids=None):

        
        if self.swap_base != "hybrid_balanced":
            for swap_label in labels:
                if swap_label not in self.swap_class_dist:
                    self.swap_class_dist[swap_label] = 1
                else:
                    self.swap_class_dist[swap_label] += 1

        #print(what_to_swap)
        if len(what_to_swap) > 0:
            if hasattr(self, "_swapper"):
                asyncio.run(self._swapper._swap(what_to_swap, labels, data_ids))
            elif hasattr(self, "_swap_queues"):
                #print("WORKER EXISTS")
                worker_id = next(self._swap_worker_queue_idx_cycle)
                self._swap_queues[worker_id].put(
                    (what_to_swap, labels, data_ids))
            self.num_swap = self.num_swap + len(what_to_swap)
        
        

    def swap_policy(self, swap_base):
        policies = {
            "entropy" : self.entropy,
            "hybrid_threshold" : self.hybrid,
            "prediction" : self.prediction,
            "random" : self.random,
            "random_fixed": self.random_fixed,
            "pure_random" : self.pure_random,
            "opposite": self.hybrid_opposite,
            "hybrid_ratio" : self.hybrid_ratio,
            "hybrid_balanced" : self.hybrid_balanced,
            "hybrid_balanced_p" : self.hybrid_balanced_p,
            #"hybrid_random" : self.hybrid_random,
            "hybrid_loss" : self.hybrid_loss,
            "all" : self.all
        }
        return policies[swap_base]

    @property
    def swap_thr(self):
        return self._swap_thr
    
    @swap_thr.setter
    def swap_thr(self, thr):
        self._swap_thr = thr

    
    @property
    def swap_loss(self):
        return self._swap_loss
    
    @swap_loss.setter
    def swap_loss(self, loss):
        self._swap_loss = loss
    
    def to_onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
        return onehot

    def get_replay_index(self, idxs, targets, data_ids=None):

        if self.rp_len is None:
            if data_ids is not None:
                return idxs, targets, data_ids
            else:
                return idxs, targets

        else:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
            
            if data_ids is not None:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs], data_ids[replay_index_of_idxs]
            else:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs]

    def prediction(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on mis-prediction
        #
        predicts = torch.max(outputs, dim=1)[1]
        selected_idx = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    def entropy(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on entropy (threshold = 1.0 : lower is easy and swap, higher is hard and preserve)
        #
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        selected_idx = (entropy.cpu() < self.threshold).squeeze().nonzero(as_tuple=True)[0]

        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])


    def hybrid(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on entropy (threshold : lower is easy and swap, higher is hard and preserve)
        #        
        #print(idxs,outputs, targets)
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        entropy_batch = (entropy.cpu() < self.threshold).squeeze()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        prediction_batch = (predicts.cpu() == targets.cpu()).squeeze()

        selected_idx = (torch.logical_and(entropy_batch, prediction_batch)).nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    #@profile
    
    def random_fixed(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        
        replay_output = outputs[replay_index_of_idxs]
        replay_idxs = idxs[replay_index_of_idxs]
        replay_targets = targets[replay_index_of_idxs]
        
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs]
        
        if data_ids is not None:
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap], replay_data_ids[:how_much_swap]
        else:
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap]
            

    def random(self, idxs, outputs, targets, data_ids=None):
        
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
        #print("replay len in batch : ", len(replay_index_of_idxs))
        #print("how much swap : ", how_much_swap)
        
        self.get_loss(replay_output, replay_targets)
        self.get_entropy(replay_output, replay_targets)
        
        selected_index = np.random.choice(len(replay_idxs), how_much_swap, replace=False)

        assert len(selected_index) == how_much_swap

        
        if data_ids is not None:
            return replay_idxs[selected_index], replay_targets[selected_index], replay_data_ids[selected_index]
        else:
            return replay_idxs[selected_index], replay_targets[selected_index]
    
    
    def pure_random(self, idxs, outputs, targets, data_ids=None):

        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        idx_to_pick = np.linspace(0, len(replay_index_of_idxs), num=how_much_swap, dtype = int, endpoint=False)
        #print("init idx_to_pick : ", idx_to_pick)

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        torch.set_printoptions(precision=4,sci_mode=False)
        
        soft_output = self.softmax(replay_output)

        torch.set_printoptions(precision=4,sci_mode=False)

        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        torch.set_printoptions(precision=4,sci_mode=False)

        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]
    
            r_range = np.where(idx_to_pick < len(r_idxs))
            sorted_r = torch.argsort(r_entropy)[idx_to_pick[r_range]]
            #print("idx_to_pick for right precition : ", idx_to_pick[r_range])

            
            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]


            if len(sorted_r) < how_much_swap:
                #print("wrong_prediction count...")
                idx_to_pick = np.delete(idx_to_pick, r_range)
                #print("idx_to_pick before subtract len : ", idx_to_pick)
                #print("len of replay samples in this batch : ", len(r_idxs))
                idx_to_pick = idx_to_pick - len(r_idxs)
                #print("idx_to_pick after subtract len : ", idx_to_pick)
                #print("\n\n")

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]

                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]
                
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]


                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]


                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)
        

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids


        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #print("selected_idxs : ", selected_idxs)
        #print("selected_targets : ", selected_targets)
        #print("\n")

        
        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets


    def hybrid_opposite(self, idxs, outputs, targets, data_ids=None):

        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)


        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #print("idxs : ",idxs)
        #print("targets : ", targets)

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("w_pred(idx) : " , w_predicted)
            #print("w_pred size : ", w_predicted.shape)

            w_idxs = replay_idxs[w_predicted]
            w_entropy = entropy[w_predicted]
            w_targets = replay_targets[w_predicted]
            if data_ids is not None:
                w_data_ids = replay_data_ids[w_predicted]

            #print("w_idxs : ", w_idxs)
            #print("w_targets : ", w_targets)
            #print("w_entropy : ", w_entropy)
            
            sorted_w = torch.argsort(w_entropy)[:how_much_swap]

            #print("sorted_w : ", sorted_w)

            selected_w_idxs = w_idxs[sorted_w]
            selected_w_targets = w_targets[sorted_w]
            if data_ids is not None:
                selected_w_data_ids = w_data_ids[sorted_w]

            #print("selected_w_idxs : ", selected_w_idxs)
            #print("selected_w_targets : ", selected_w_targets)

            if len(sorted_w) < how_much_swap:

                r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                r_idxs = replay_idxs[r_predicted]
                r_entropy = entropy[r_predicted]
                r_targets = replay_targets[r_predicted]
                    
                if data_ids is not None:
                    r_data_ids = replay_data_ids[r_predicted]

                #print("r_pred(idx) : " , r_predicted)
                #print("r_idxs : ", r_idxs)
                #print("r_targets : ", r_targets)
                #print("r_entropy : ", r_entropy)
                
                sorted_r = torch.argsort(r_entropy)[-(how_much_swap-len(sorted_w)):]

                #print("sorted_r : ", sorted_r)

                selected_r_idxs = r_idxs[sorted_r]
                selected_r_targets = r_targets[sorted_r]
                    
                if data_ids is not None:
                    selected_r_data_ids = r_data_ids[sorted_r]

                    
                #print("selected_r_idxs : ", selected_r_idxs)
                #print("selected_r_targets : ", selected_r_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_w_idxs
                selected_targets = selected_w_targets
                if data_ids is not None:
                    selected_data_ids = selected_w_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #print("selected_idxs : ", selected_idxs)
        #print("selected_targets : ", selected_targets)
        #print("\n")
        
        assert len(selected_idxs) == how_much_swap


        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def hybrid_ratio(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #print("total batch len : ", len(idxs))
        #print("replay batch len : ", replay_index_of_idxs)
        #print("how_much_swap : ", how_much_swap)


        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            
            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            #print("sorted_r : ", sorted_r)

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]

                #print("sorted_w : ", sorted_w)

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        #print("selected_targets : ", selected_targets)
        #print("\n")

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def get_entropy(self, outputs, targets):
        
        if self._get_entropy == False:
            return

        print("GET ENTROPY IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_entropy = entropy[r_predicted]
            
        self.data_correct_entropy.extend(r_entropy.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_entropy = entropy[w_predicted]
            
        self.data_wrong_entropy.extend(w_entropy.tolist())


    def get_loss(self, outputs, targets):

        if self._get_loss == False:
            return
        
        print("GET LOSS IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        try:
            loss = self.swap_loss(outputs, targets)
        except ValueError:
            #print(outputs.shape)
            targets_one_hot = self.to_onehot(targets, outputs.shape[1])
            loss = self.swap_loss(outputs, targets_one_hot)

            loss = loss.view(loss.size(0), -1)
            loss = loss.mean(-1)
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]

        #print(loss.shape, outputs.shape, predicts.shape)

        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_loss = loss[r_predicted]
            
        self.data_correct_loss.extend(r_loss.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_loss = loss[w_predicted]
            
        self.data_wrong_loss.extend(w_loss.tolist())


    def hybrid_loss(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        #print("total batch len : ", len(idxs))
        #print("replay batch len : ", replay_index_of_idxs)
        #print("how_much_swap : ", how_much_swap)


        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        
        self.get_loss(replay_output, replay_targets)
        self.get_entropy(replay_output, replay_targets)
        

        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        #print("BEFORE select batch_dist : ", batch_dist)
        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected
        #print(total_how_much_swap, expected)

        #print(shortage)

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        #print("BEFORE : ", batch_dist)

        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)
            
        #print("batch_dist {k : (decimal, how_much_swap_for_this_class)} : ", batch_dist)
        #print("total swap num : ", total_how_much_swap)
        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):
            #print("current class.. ", k)
            #print("how much swap in current class... ", int(v[1]))
            how_much_swap = int(v[1])

            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            
            try:
                loss = self.swap_loss(outputs, targets).cpu()
            except ValueError:
                #print(outputs.shape)
                targets_one_hot = self.to_onehot(targets, outputs.shape[1])
                loss = self.swap_loss(outputs, targets_one_hot).cpu()
                loss = loss.view(loss.size(0), -1)
                loss = loss.mean(-1)
            
            #print("LOSS : ", loss, loss.shape)
            #print("replay_index_of_idxs : ", replay_index_of_idxs)
            #print("replay idxs : ", replay_idxs)
            #print("replay_targets : ", replay_targets)
            #print("entropy : ", entropy)
            #print("entropy size : ", entropy.shape)
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = cur_replay_idxs[r_predicted]
            r_loss = loss[r_predicted]            
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_loss : ", r_loss)
            
            sorted_r = torch.argsort(r_loss)[:how_much_swap]
            #print("how_much_swap : ", how_much_swap)
            #print("sorted_r : ", sorted_r)

            #to check this code validate
            selected_r_loss = r_loss[sorted_r]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_loss = loss[w_predicted]

                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_loss : ", w_loss.shape)

                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_loss : ", w_loss)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                #######
                sorted_w = torch.argsort(w_loss)[:w_how_much_swap]

                #print("sorted_w : ", sorted_w)
                #######
                ####### sorted_w = torch.argsort(w_entropy)[:w_how_much_swap]
                #######
                ####### sorted_w = np.random.choice(len(w_entropy), w_how_much_swap, replace=False)
                
                #print("sorted_w : ", sorted_w)

                #to check this code validate
                selected_w_loss = w_loss[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids

        
            if i==0:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

            else:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

        
        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets


    def hybrid_balanced_p(self, idxs, outputs, targets, data_ids=None):
        
        #print("\n")
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))


        #print("total_how_much_swap : ", total_how_much_swap)


        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
        
        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        #print("BEFORE : ", batch_dist)
        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected
        #print("shortage : ", shortage)

        #print(shortage)

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            #p = [w[0] for (_,w) in get_dec_samples]
            #l = [k for (k,_) in get_dec_samples]
            
            #selected_dec_samples = np.random.choice(l, size=shortage, replace=False, p=[i/sum(p) for i in p] ) #balanced_p_v2

            
            #print(selected_dec_samples)

            for t in selected_dec_samples:
                k, _ = t
                #k = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        #print("AFTER : ", batch_dist)

        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):

            #print("current class : ", k)
            how_much_swap = int(v[1])
            if how_much_swap == 0:
                continue
            
            
            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            soft_output = self.softmax(cur_replay_output)
            entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
            #print("cur_cls_idx : ", cur_cls_idx)
            #print("cur_replay_idxs : ", cur_replay_idxs)
            #print("cur_replay_targets : ", cur_replay_targets)
            #print("entropy : ", entropy)
            #print("entropy size : ", entropy.shape)
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            #print("prediction : ", predicts)
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = cur_replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            sorted_r = torch.argsort(r_entropy)[:how_much_swap]


            #to check this code validate
            selected_r_entropy = r_entropy[sorted_r]


            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                #######
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]
                #######
                ####### sorted_w = torch.argsort(w_entropy)[:w_how_much_swap]
                #######
                ####### sorted_w = np.random.choice(len(w_entropy), w_how_much_swap, replace=False)
                
                #print("sorted_w : ", sorted_w)

                #to check this code validate
                selected_w_entropy = w_entropy[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids

            try:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

            except:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets



    def hybrid_balanced(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #print("total batch len : ", len(idxs))
        #print("replay batch len : ", replay_index_of_idxs)
        #print("how_much_swap : ", how_much_swap)


        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            
            sorted_r_org = torch.argsort(r_entropy)

            selected = []
            filled_counter = 0

            for i, idx in enumerate(sorted_r_org):

                if filled_counter >= how_much_swap:
                    break

                label = r_targets[idx].item()
                if label in self.swap_class_dist:
                    if self.swap_class_dist[label] + 1 <= self.swap_thr:
                        self.swap_class_dist[label] += 1
                        filled_counter +=1
                        selected.append(i)
                    else:
                        continue
                else:
                    self.swap_class_dist[label] = 1
                    filled_counter +=1
                    selected.append(i)
                #print("LABEL : ", label)
                #print("class dist : ", self.swap_class_dist)
                #print("filled_count, how_much_swap : ", filled_counter, how_much_swap)

            sorted_r = sorted_r_org[selected][:how_much_swap]

            #print("sorted_r : ", sorted_r)

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)



            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w_org = torch.argsort(w_entropy, descending=True)


                selected = []
                filled_counter = 0

                for i, idx in enumerate(sorted_w_org):
                    if filled_counter >= w_how_much_swap:
                        break
                    label = w_targets[idx].item()
                    if label in self.swap_class_dist:
                        if self.swap_class_dist[label] + 1 <= self.swap_thr:
                            self.swap_class_dist[label] += 1
                            filled_counter +=1
                            selected.append(i)
                        else:
                            continue
                    else:
                        self.swap_class_dist[label] = 1
                        filled_counter +=1
                        selected.append(i)
                    
                    #print("LABEL : ", label)
                    #print("class dist : ", self.swap_class_dist)
                    #print("filled_count, how_much_swap : ", filled_counter, w_how_much_swap)

                sorted_w = sorted_w_org[selected][:w_how_much_swap]
                    
                #print("sorted_w : ", sorted_w)

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        if len(selected_idxs) != how_much_swap:
            print("ADDITIONAL SWAP CAND SELECTION!!!")
            for unselected in sorted_r_org:
                
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_r:
                    print(selected_idxs)
                    print(r_idxs[unselected])
                    selected_idxs = torch.cat((selected_idxs,r_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,r_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,r_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[r_targets[unselected].item()] += 1
            
            for unselected in sorted_w_org:
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_w:
                    print(selected_idxs)
                    print(w_idxs[unselected])
                    selected_idxs = torch.cat((selected_idxs,w_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,w_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,w_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[w_targets[unselected].item()] += 1


        assert len(selected_idxs) == how_much_swap

        #print("selected_targets : ", selected_targets)
        #print("\n")

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    
    #@profile
    def all(self, idxs, outputs=None, targets=None, data_ids=None):
    
        ######### changed for time measurement
        if targets is not None and data_ids is not None:   
            return self.get_replay_index(idxs, targets, data_ids)
        else:
            return self.get_replay_index(idxs, targets)
            

    def hybrid_random(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        #print("HOW MUCH SWAP ? ", how_much_swap)

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            
            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            #print("sorted_r : ", sorted_r)

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)

                idx_to_pick = np.linspace(0, len(w_idxs), num=w_how_much_swap, dtype = int, endpoint=False)
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]

                #print("sorted_w : ", sorted_w)

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        #print("selected_targets : ", selected_targets)
        #print("\n")

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets
