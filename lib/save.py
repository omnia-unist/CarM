import os
from array import array
import asyncio
import pickle

import queue
import threading
import multiprocessing as python_multiprocessing

class DataSaver(object):
    #m = python_multiprocessing.Manager()
    #print('MANAGER_2 PID:', m._process.ident)

    #num_file_for_label = m.dict()
    #num_file_for_label = dict()
    #observed = set()

    def __init__(self, rb_path, store_budget=None):
        if len(rb_path)-1 == '/':
            rb_path = rb_path[:len(rb_path)-1]
        self.rb_path = rb_path
        self.store_budget = store_budget
        print("STORAGE_BUDGET : ", self.store_budget)


        m = python_multiprocessing.Manager()
        print('MANAGER_2 PID:', m._process.ident)

        self.num_file_for_label = m.dict()

        print("\n==================\nSAVE WORKER IS CREATED")
        print("num_file_for_label : ", self.num_file_for_label)
    
    def before_train(self):
        print("SAVE WORKER CREATED!")
        self.save_done_event = python_multiprocessing.Event()
        self.save_queue = python_multiprocessing.Queue()
        self.save_worker = python_multiprocessing.Process(
                    target = self.save_loop,
                    args=(self.save_queue, self.save_done_event)
        )
        self.save_worker.daemon = True
        self.save_worker.start()

    def after_train(self):
        self.save_done_event.set()
        self.save_queue.put((None,None,None))
        self.save_worker.join()
        self.save_queue.cancel_join_thread()
        self.save_queue.close()

        print("SAVE SHUTDOWN")

    async def makedir(self, path):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except:
            return False
    
    async def data_save(self, img, path, label, logit=None, logit_path=None):
        try:
            #print(logit_path)
            img.save(path)
            if logit is not None and logit_path is not None:
                with open(logit_path, 'wb') as f:
                    pickle.dump(logit, f)
                #f.close()
            return label
        
        except Exception as e:
            print(e)
            return False


    async def main_budget(self, stream_data, stream_targets, stream_outputs=None):

        for i, (data, label) in enumerate(zip(stream_data, stream_targets)):        


            #print("BEFORE SAVE")
            label_path = self.rb_path + '/' + str(label)
            if label not in self.num_file_for_label:
                self.num_file_for_label.update( {label : 0} )
                await self.makedir(label_path)
            
            storage_per_cls = self.store_budget // len(self.num_file_for_label)

            for la in self.num_file_for_label.keys():
                del_label_path = self.rb_path + '/' + str(la)
                if storage_per_cls <= self.num_file_for_label[la]:
                    #print("STORAGE_PER_CLS : ", storage_per_cls)
                    del_st = storage_per_cls
                    del_en = self.num_file_for_label[la]

                    for del_file in range(del_en, del_st-1,-1):
                        del_filepath = del_label_path + '/' + str(del_file) + '.png' 
                        os.remove(del_filepath)
                        self.num_file_for_label[la] = self.num_file_for_label[la] - 1
                    
                    
            if storage_per_cls == self.num_file_for_label[label]:
                del_filepath = label_path + '/' + str(self.num_file_for_label[label]) + '.png'
                os.remove(del_filepath)
                self.num_file_for_label[label] = self.num_file_for_label[label] - 1

            assert storage_per_cls > self.num_file_for_label[label]

            curr_num = str(self.num_file_for_label[label] + 1)

            file_name = curr_num + ".png"
            file_path = label_path + '/' + file_name

            #print(file_path)

            if stream_outputs is not None:
                logit_file_name = curr_num + ".pkl"
                logit_file_path = label_path + '/' + logit_file_name
                logit = stream_outputs[i]
                completed_label = await self.data_save(data, file_path, label, logit, logit_file_path)
                if completed_label is not False:
                    self.num_file_for_label[completed_label] += 1
            else:
                completed_label = await self.data_save(data, file_path, label)
                if completed_label is not False:
                    self.num_file_for_label[completed_label] += 1
            #print("AFTER SAVE")
            del data, label
        #print("DATA ARE ALL SAVED! ", self.num_file_for_label)


    async def main(self, stream_data, stream_targets, stream_outputs=None):
        for i, (data, label) in enumerate(zip(stream_data, stream_targets)):
            #print("BEFORE SAVE")
            label_path = self.rb_path + '/' + str(label)
            if label not in self.num_file_for_label:
                self.num_file_for_label.update( {label : 0} )
                await self.makedir(label_path)
            
            curr_num = str(self.num_file_for_label[label] + 1)

            file_name = curr_num + ".png"
            file_path = label_path + '/' + file_name

            #print(file_path)

            if stream_outputs is not None:                
                logit_file_name = curr_num + ".pkl"
                logit_file_path = label_path + '/' + logit_file_name
                logit = stream_outputs[i]
                completed_label = await self.data_save(data, file_path, label, logit, logit_file_path)
                if completed_label is not False:
                    self.num_file_for_label[completed_label] += 1
            else:
                completed_label = await self.data_save(data, file_path, label)
                if completed_label is not False:
                    self.num_file_for_label[completed_label] += 1
            #print("AFTER SAVE")
            del data, label
        #print("DATA ARE ALL SAVED! ", self.num_file_for_label)
    
    
    def save_loop(self, save_queue, save_done_event):
        print("SAVE WORKER ID : ", os.getpid())
        while True:
            stream_data, stream_targets, stream_outputs = self.save_queue.get()
            if stream_data == None:
                assert save_done_event.is_set()
                break
            if save_done_event.is_set():
                continue

            if self.store_budget is not None:
                asyncio.run(self.main_budget(stream_data, stream_targets, stream_outputs))
            else:
                asyncio.run(self.main(stream_data, stream_targets, stream_outputs))
            
            del stream_data, stream_targets, stream_outputs
            
    def save(self, stream_data, stream_targets, stream_outputs=None):
        self.save_queue.put((stream_data, stream_targets, stream_outputs))

        """        
        if self.imagenet == True:
            print("SAVING ALL STREAM SAMPLES...")
            asyncio.run(self.main(stream_data, stream_targets, stream_outputs))
        else:
            #print("SAVING ALL STREAM SAMPLES...WORKER")
            self.save_queue.put((stream_data, stream_targets, stream_outputs))
        """