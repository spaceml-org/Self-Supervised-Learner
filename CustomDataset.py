###CustomDataset.py

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import PIL
import os
from fnmatch import fnmatch
import numpy as np

import pytorch_lightning as pl
import splitfolders
from torchvision.datasets import ImageFolder, DatasetFolder
from os import path
from torch.utils.data import DataLoader
import shutil

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
    
    
class FolderDataset_helper(Dataset):


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
    
class FolderDataset2(pl.LightningDataModule):
    
    def __init__(self, DATA_PATH, val_split, train_transform = None, val_transform = None):
        super().__init__()
        self.DATA_PATH = DATA_PATH
        self.val_split = val_split
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        self.num_workers = 2
        self.batch_size = 64
        
    def setup(self):
        shutil.rmtree('split_data', ignore_errors=True)
        if not (path.isdir(f"{self.DATA_PATH}/train") and path.isdir(f"{self.DATA_PATH}/val")): 
            splitfolders.ratio(self.DATA_PATH, output=f"split_data", ratio=(1-self.val_split, self.val_split), seed = 10)
        def loader(data_path):
            return PIL.Image.open(data_path).convert('RGB')
        
        self.finetune_dataset = DatasetFolder(f"split_data/train/", transform = self.train_transform, extensions = '.tif', loader = loader)
        self.finetune_val_dataset = DatasetFolder(f"split_data/val/", transform = self.val_transform,  extensions = '.tif', loader = loader)
        print(f'Loaded {len(self.finetune_dataset)} images for training..')
#         FolderDataset_helper(self.DATA_PATH, validation = False, 
#                               val_split = self.val_split, 
#                               withold_train_percent = 0, 
#                               transform = self.train_transform, 
#                               image_type = 'tif'
#                               ) 
#         self.finetune_val_dataset = FolderDataset_helper(self.DATA_PATH, validation = True, 
#                               val_split = self.val_split, 
#                               withold_train_percent = 0, 
#                               transform = self.val_transform, 
#                               image_type = 'tif'
#                               )
        self.num_samples = len(self.finetune_dataset)
        self.num_classes = len(self.finetune_dataset.classes)
     
    def train_dataloader(self):
        return DataLoader(self.finetune_dataset, batch_size=self.batch_size, drop_last = True, num_workers=self.num_workers)

    def val_dataloader(self):
        try:
            return DataLoader(self.finetune_val_dataset, batch_size=self.batch_size, drop_last = True, num_workers=self.num_workers)
        except:
            return None

