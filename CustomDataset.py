###CustomDataset.py

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import PIL
import os
from fnmatch import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import splitfolders
from torchvision.datasets import ImageFolder, DatasetFolder
from os import path
from torch.utils.data import DataLoader
import shutil
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class FolderDataset(Dataset):


    def __init__(self, DATA_PATH, validation = False, val_split = 0.2, transform=None, withold_train_percent = 0, image_type = 'tif'):
        """
        Args:
            DATA_PATH (string): path to folder containing images that can be read by the PIL library (most image types)
            validation (bool): whether to return validation data or training data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        #The following code goes through the directories in DATA_PATH and pulls the labels and image files from the directory
        #The paths to the image are stored in a list of directories (self.dirs) and their respective labels (self.labels)
        self.image_type = image_type
        pattern = f"*.{self.image_type}" #pattern to find

        self.dirs= []
        self.labels = []
        for path, subdirs, files in os.walk(DATA_PATH):
            for name in files:
                if fnmatch(name, pattern):
                    self.dirs.append(os.path.join(path, name))
                    self.labels.append(path.split('/')[-1])
                    
        self.transform = transform
       

        #use sklearn's module to return training data and test data
        self.mydict={}
        i = 0
        for item in self.labels:
            if(i>0 and item in self.mydict):
                continue
            else:    
                i = i+1
                self.mydict[item] = i

        k=[]
        for item in self.labels:
            k.append(self.mydict[item])

        self.labels = np.array(k)-1

        if validation:
            _, self.dirs, _, self.labels = train_test_split(self.dirs, self.labels, test_size = val_split, random_state=42)

        else:
            self.dirs, _, self.labels, _ = train_test_split(self.dirs, self.labels, test_size = val_split, random_state=42)

        #witholding data for experimentation
        np.random.seed(0)
        indxs = np.random.choice(len(self.dirs), int(len(self.dirs)*(1-withold_train_percent))) 
        temp_dirs = []
        temp_labels = []
        for ix in indxs:
          temp_dirs.append(self.dirs[ix])
          temp_labels.append(self.labels[ix])

        self.dirs = temp_dirs
        self.labels = temp_labels
        print('DICTIONARY FOR LABELS: ')
        print(self.mydict)
          
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        im = PIL.Image.open(self.dirs[idx])
        im = im.convert('RGB')
        sample = self.transform(im)
        #doing the PIL.image.open and transform stuff here is quite slow
        return (sample, self.labels[idx]) # we dont need ylabel except for validating our unsupervised learning.

class ImageModule(pl.LightningDataModule):
    
    def __init__(self, DATA_PATH, val_split, train_transform = None, val_transform = None, num_workers = 0):
        super().__init__()
        self.DATA_PATH = DATA_PATH
        self.val_split = val_split
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        self.num_workers = num_workers
        self.batch_size = 64
        
    def setup(self):
        shutil.rmtree('split_data', ignore_errors=True)
        if not (path.isdir(f"{self.DATA_PATH}/train") and path.isdir(f"{self.DATA_PATH}/val")): 
            splitfolders.ratio(self.DATA_PATH, output=f"split_data", ratio=(1-self.val_split, self.val_split), seed = 10)
            
        self.finetune_dataset = ImageFolder('split_data/train', transform = self.train_transform)
        self.finetune_val_dataset = ImageFolder('split_data/val', transform = self.val_transform)

        self.num_samples = len(self.finetune_dataset)
        self.num_classes = len(self.finetune_dataset.classes)
     
    def train_dataloader(self):
        return DataLoader(self.finetune_dataset, batch_size=self.batch_size, drop_last = True, num_workers=self.num_workers, shuffle = True)

    def val_dataloader(self):
        try:
            return DataLoader(self.finetune_val_dataset, batch_size=self.batch_size, drop_last = True, num_workers=self.num_workers)
        except:
            return None
        
class DaliModule(pl.LightningDataModule):
    
    def __init__(self, DATA_PATH, val_split, input_height, batch_size, train_transform, val_transform):
        super().__init__()
        self.DATA_PATH = DATA_PATH
        self.val_split = val_split
        self.batch_size = batch_size
        self.input_height = input_height
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self):
        shutil.rmtree('split_data', ignore_errors=True)
        if not (path.isdir(f"{self.DATA_PATH}/train") and path.isdir(f"{self.DATA_PATH}/val")): 
            splitfolders.ratio(self.DATA_PATH, output=f"split_data", ratio=(1-self.val_split, self.val_split), seed = 10)
            self.DATA_PATH = 'split_data'

        print('Working from data directory: ', self.DATA_PATH)

        self.train_pipe = self.train_transform(f'{self.DATA_PATH}/train', batch_size = self.batch_size, input_height = self.input_height, num_threads = 4, device_id = 0)
        self.val_pipe = self.val_transform(f'{self.DATA_PATH}/val', batch_size = self.batch_size, input_height = self.input_height, num_threads = 4, device_id = 0)

        #We have this wrapper to allow for modification if the transform is different than expected and to return the proper length for the dataset for the pl.Trainer, which is an iterable with default length 0
        class LightningSSLWrapper(DALIGenericIterator):
            def __init__(self, num_samples, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)
                self.num_samples = num_samples

            def __next__(self):
                out = super().__next__()
                out = out[0]
                return [out[k] for k in self.output_map[:-1]], out['label']

            def __len__(self):
              return self.num_samples//self.batch_size

        class LightningFTWrapper(LightningSSLWrapper):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out, label = super().__next__()
                label = torch.squeeze(label)
                return out[0], label.long()

        train_labels = [f'im{i}' for i in range(1,self.train_pipe.COPIES+1)]
        train_labels.append('label')

        val_labels = [f'im{i}' for i in range(1,self.val_pipe.COPIES+1)]
        val_labels.append('label')

        size_train = sum([len(files) for r, d, files in os.walk(f'{self.DATA_PATH}/train')])
        size_val =  sum([len(files) for r, d, files in os.walk(f'{self.DATA_PATH}/val')])

        if self.train_pipe.COPIES > 1:
            self.train_loader = LightningSSLWrapper(size_train, self.train_pipe, train_labels, auto_reset=True, last_batch_policy = LastBatchPolicy.DROP)
            self.val_loader = LightningSSLWrapper(size_val, self.val_pipe, val_labels, auto_reset=True, last_batch_policy = LastBatchPolicy.DROP)
        else:
            self.train_loader = LightningFTWrapper(size_train, self.train_pipe, train_labels, auto_reset=True, last_batch_policy = LastBatchPolicy.DROP)
            self.val_loader = LightningFTWrapper(size_val, self.val_pipe, val_labels, auto_reset=True, last_batch_policy = LastBatchPolicy.DROP)

        self.num_samples = sum([len(files) for r, d, files in os.walk(f'{self.DATA_PATH}/train')])
        self.num_classes = len([i for i in os.listdir(f'{self.DATA_PATH}/train') if os.path.isdir(f'{self.DATA_PATH}/train/{i}')])

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        try:
            return self.val_loader 
        except:
            return None

