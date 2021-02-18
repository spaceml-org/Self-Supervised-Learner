import os
from termcolor import colored
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

from torch.optim import Adam
import pytorch_lightning as pl
import numpy as np
import math

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.simclr_module import Projection

#Internal Imports
from dali_utils.dali_transforms import SimCLRTransform
from dali_utils.lightning_compat import SimCLRWrapper

class SIMCLR(SimCLR):

    def __init__(self, encoder, DATA_PATH, VAL_PATH, hidden_dims, image_size, transform = SimCLRTransform, **simclr_hparams):

        self.DATA_PATH = DATA_PATH
        self.VAL_PATH = VAL_PATH
        self.hidden_dims = hidden_dims
        self.transform = transform
        self.image_size = image_size
        self.simclr_hparams = simclr_hparams
        self.num_image_copies = 3

        super().__init__(self.simclr_hparams)
        self.encoder = encoder
            
        self.projection = Projection(input_dim = self.encoder.embedding_size, hidden_dim = self.hidden_dims)

        self.save_hyperparameters()
  
    #override pytorch SIMCLR with our own encoder so we will overwrite the function plbolts calls to init the encoder
    def init_model(self):
        return None

    def setup(self, stage = 'train'):
        if stage == 'train':
            self.train_loader = SimCLRWrapper(self.DATA_PATH, transform = self.transform(batch_size = self.batch_size, image_size = self.image_size, copies = 3, mode = 'train'))
            self.val_loader = SimCLRWrapper(self.VAL_PATH, transform = self.transform(batch_size = self.batch_size, image_size = self.image_size, copies = 3, mode = 'validation'))
        elif stage == 'test' or 'inference':
            self.test_dataloader = SimCLRWrapper(self.DATA_PATH, transform = self.transform(batch_size = self.batch_size, image_size = self.image_size, copies = 1, mode = 'inference'))
            self.inference_dataloader = self.test_dataloader
     
    def train_dataloader(self):
        return self.train_loader
  
    def val_dataloader(self):
        return self.val_loader

 
