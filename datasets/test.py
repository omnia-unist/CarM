from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder

from PIL import Image
import numpy as np
import os
import pickle
import torch

def get_test_set(test_set_name, data_manager, test_transform):
    print(test_set_name)
    test_set = {
        "imagenet" : ImagenetTestDataset,
        "imagenet100" : ImagenetTestDataset,
        "imagenet1000" : ImagenetTestDataset,
        "tiny_imagenet" : TinyImagenetTestDataset,
        "cifar100" : Cifar100TestDataset,
        "mini_imagenet" : MiniImagenetTestDataset,
        "cifar10" : Cifar10TestDataset,
    }
    if test_set == "imagenet100":
        return ImagenetTestDataset(data_manager=data_manager, test_transform=test_transform, num_class=100)
    else:
        return test_set[test_set_name](data_manager=data_manager, test_transform=test_transform)


class ImagenetTestDataset(Dataset):
    def __init__(self,
                 root='/data/Imagenet',
                 #root='/data',
                 data_manager=None,
                 split='val',
                 test_transform=None,
                 target_transform=None,
                 num_class=1000
                 ):
                 
        self.data_manager = data_manager
        self.test_transform = test_transform

        self.num_class = num_class

        if self.num_class == 1000:
            self.data_paths, self.labels = self.load_data('data/imagenet-1000/val.txt')
        elif self.num_class == 100:
            self.data_paths, self.labels = self.load_data('data/imagenet-100/val.txt')

        self.data = list()
        self.targets = list()

    def load_data(self, fpath):
        data = []
        labels = []

        lines = open(fpath)
        
        for i in range(self.num_class):
            data.append([])
            labels.append([])

        for line in lines:
            arr = line.strip().split()
            data[int(arr[1])].append(arr[0])
            labels[int(arr[1])].append(int(arr[1]))

        return data, labels

    def append_task_dataset(self, task_id):
        print("data_manager.classes_per_task[task_id] : ", self.data_manager.classes_per_task[task_id])
        for label in self.data_manager.classes_per_task[task_id]:
            actual_label = self.data_manager.map_int_label_to_str_label[label]

            if label in self.targets:
                continue
            for data_path in self.data_paths[actual_label]:
                data_path = os.path.join('/data/Imagenet', data_path)
                with open(data_path,'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                self.data.append(img)
                self.targets.append(label)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class TinyImagenetTestDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str='/data', train: bool=False, 
                 data_manager=None, test_transform: transforms=None,
                 download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = '/data'
        self.train = train
        self.download = download
        self.data_manager = data_manager
        self.test_transform = test_transform

        if download:
            from google_drive_downloader import GoogleDriveDownloader as gdd
            # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
            print('Downloading dataset')
            gdd.download_file_from_google_drive(
                file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

        self.TestData = []
        self.TestLabels = []
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            print("the size of test set is %s"%(str(self.TestData.shape)))
            print("the size of test label is %s"%str(self.TestLabels.shape))
        print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        img, target = Image.fromarray(np.uint8(255 *self.TestData[index])), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]


class MiniImagenetTestDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self,root='/data',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True) -> None:
        self.root = '/data'
        self.train = train
        self.data_manager = data_manager
        self.test_transform = test_transform
        
        self.data = []
        self.targets = []

        self.TestData = []
        self.TestLabels = []

        train_in = open(root+"/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open(root+"/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open(root+"/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))

        TEST_SPLIT = 1 / 6

        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            x_test = cur_x[: int(600 * TEST_SPLIT)]
            y_test = cur_y[: int(600 * TEST_SPLIT)]
            test_data.append(x_test)
            test_label.append(y_test)

        self.data = np.concatenate(test_data)
        self.targets = np.concatenate(test_label)
        self.targets = torch.from_numpy(self.targets).type(torch.LongTensor)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            #print("the size of test set is %s"%(str(self.TestData.shape)))
            #print("the size of test label is %s"%str(self.TestLabels.shape))
        #print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]

class Cifar100TestDataset(CIFAR100):
    def __init__(self,root='./data',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super().__init__(root,train=train,
                        transform=test_transform,
                        target_transform=target_transform,
                        download=download)

        self.TestData = []
        self.TestLabels = []
        self.data_manager = data_manager
        self.test_transform = test_transform

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            #print("the size of test set is %s"%(str(self.TestData.shape)))
            #print("the size of test label is %s"%str(self.TestLabels.shape))
        #print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]



class Cifar10TestDataset(CIFAR10):
    def __init__(self,root='./data',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super().__init__(root,train=train,
                        transform=test_transform,
                        target_transform=target_transform,
                        download=download)

        self.TestData = []
        self.TestLabels = []
        self.data_manager = data_manager
        self.test_transform = test_transform
        print("test_transform : ", self.test_transform)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        print("data_manager.classes_per_task[task_id] : ", self.data_manager.classes_per_task[task_id])
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            print("the size of test set is %s"%(str(self.TestData.shape)))
            print("the size of test label is %s"%str(self.TestLabels.shape))
        print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]