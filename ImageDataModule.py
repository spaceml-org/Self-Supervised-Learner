import pytorch_lightning as pl
import splitfolders
from torchvision.datasets import ImageFolder
from os import path
from torch.utils.data import DataLoader
import shutil

class ImageDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size = 64, train_transform = None, val_transform = None, val_split = 0.2):
        super().__init__()
        self.batch_size = batch_size
        self.PATH = data_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.val_split = val_split
        
    def setup(self, stage=None):
        shutil.rmtree('split_data', ignore_errors=True)
        if not (path.isdir(f"{self.PATH}/train") and path.isdir(f"{self.PATH}/validation")): 
            splitfolders.ratio(self.PATH, output=f"split_data", ratio=(1-self.val_split, self.val_split), seed = 10)
            self.train = ImageFolder('split_data/train', transform = self.train_transform)
            if self.val_split > 0:
                self.val = ImageFolder('split_data/val', transform = self.val_transform)
        else:
            self.train = ImageFolder(f'{self.PATH}/train', transform = self.train_transform)
            if self.val_split > 0:
                self.val = ImageFolder(f'{self.PATH}/validation', transform = self.val_transform)
        self.num_classes = len(self.train.classes)
        self.num_samples = len(self.train)
        print('We have the following classes: ', self.train.classes)
        

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last = True)

    def val_dataloader(self):
        try:
            return DataLoader(self.val, batch_size=self.batch_size, drop_last = True)
        except:
            return None
