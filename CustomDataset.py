###CustomDataset.py

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import PIL
import os
from fnmatch import fnmatch
import numpy as np

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
          
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        im = PIL.Image.open(self.dirs[idx])
        sample = self.transform(im)
        #doing the PIL.image.open and transform stuff here is quite slow
        return (sample, self.labels[idx]) # we dont need ylabel except for validating our unsupervised learning.
