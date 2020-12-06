import pytorch_lightning as pl
import splitfolders
from torchvision.datasets import ImageFolder
from os import path
from torch.utils.data import DataLoader

class ImageDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size = 64, transforms = None):
        super().__init__()
        self.batch_size = batch_size
        self.PATH = data_dir
        self.transform = transforms
        
    def setup(self, stage=None):
        if not (path.isdir(f"{self.PATH}/train") and path.isdir(f"{self.PATH}/validation")): 
            splitfolders.ratio(self.PATH, output=f"split_data", ratio=(.8, .2), seed = 10)
            self.train = ImageFolder('split_data/train', transform = self.transform)
            self.val = ImageFolder('/split_data/val', transform = self.transform)
        else:
            self.train = ImageFolder(f'{self.PATH}/train', transform = self.transform)
            self.val = ImageFolder(f'{self.PATH}/validation', transform = self.transform)
        self.num_classes = len(self.train.classes)
        self.num_samples = len(self.train)
        print('We have the following classes: ', self.train.classes)
        

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last = True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last = True)
