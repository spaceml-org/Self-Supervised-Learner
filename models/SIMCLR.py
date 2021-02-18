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

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator

from pl_bolts.models.self_supervised import SimCLR

#Internal Imports
from .dali_utils.dali_transforms import SimCLRTrainDataTransform, SimCLRValDataTransform
from .dali_utils.setup import setup_dali

class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

class SIMCLR(SimCLR):

    def __init__(self, encoder, DATA_PATH, VAL_PATH, hidden_dims, image_size, train_transform = SimCLRTrainDataTransform, val_transform = SimCLRValDataTransform, **simclr_hparams):

        self.DATA_PATH = DATA_PATH
        self.VAL_PATH = VAL_PATH
        self.hidden_dims = hidden_dims
        self.train_transform = train_transform
        self.val_transform = val_transform
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

    def setup(self, stage):
        if stage == 'train':
            self.train_loader = setup_dali(self.DATA_PATH, transform = self.train_transform(batch_size = self.batch_size, image_size = self.image_size, copies = 3, labels = True))
            self.val_loader = setup_dali(self.VAL_PATH, transform = self.val_transform(batch_size = self.batch_size, image_size = self.image_size, copies = 3, labels = True))
        elif stage == 'test' or 'inference':
            self.test_dataloader = setup_dali(self.DATA_PATH, transform = self.val_transform(batch_size = self.batch_size, image_size = self.image_size, copies = 1, labels = False))
            self.inference_dataloader = self.test_dataloader
     
    def train_dataloader(self):
        return self.train_loader
  
    def val_dataloader(self):
        return self.val_loader

 
