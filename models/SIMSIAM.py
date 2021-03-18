import os
from termcolor import colored
import numpy as np
import math
from argparse import ArgumentParser
from termcolor import colored
from enum import Enum  

import torch
from torch.nn import functional as F
from torch import nn
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SimSiam
from pl_bolts.models.self_supervised.simsiam.models import SiameseArm
#from pl_bolts.models.self_supervised.simclr.simclr_module import Projection

#Internal Imports
from dali_utils.dali_transforms import SimCLRTransform #same transform as SimCLR
from dali_utils.lightning_compat import SimCLRWrapper

class SIMSIAM(SimSiam):

    def __init__(self, encoder, DATA_PATH, VAL_PATH, hidden_dim, image_size, seed, cpus, transform = SimCLRTransform, **simsiam_hparams):
        
        data_temp = ImageFolder(DATA_PATH)
        
        #derived values (not passed in) need to be added to model hparams
        simsiam_hparams['num_samples'] = len(data_temp)
        simsiam_hparams['dataset'] = None
        simsiam_hparams['max_epochs'] = simsiam_hparams['epochs']
        
        self.DATA_PATH = DATA_PATH
        self.VAL_PATH = VAL_PATH
        self.hidden_dim = hidden_dim
        self.transform = transform
        self.image_size = image_size
        self.cpus = cpus
        self.seed = seed
        
        
        super().__init__(**simsiam_hparams)
        
        #overriding pl_lightning encoder after original simsiam init
        
        self.online_network = SiameseArm(
            encoder, input_dim=encoder.embedding_size, hidden_size=self.hidden_dim, output_dim=self.feat_dim
        )
        self.online_network.emedding_size = encoder.embedding_size
        self.encoder = self.online_network
        
        self.save_hyperparameters()
        
        
        
  
    #override pytorch SimSiam with our own encoder so we will overwrite the function plbolts calls to init the encoder
    def init_model(self):
        return None


    def setup(self, stage = 'inference'):
        Options = Enum('Loader', 'fit test inference')
        if stage == Options.fit.name:
            train = self.transform(self.DATA_PATH, batch_size = self.batch_size, input_height = self.image_size, copies = 3, stage = 'train', num_threads = self.cpus, device_id = self.local_rank, seed = self.seed)
            val = self.transform(self.VAL_PATH, batch_size = self.batch_size, input_height = self.image_size, copies = 3, stage = 'validation', num_threads = self.cpus, device_id = self.local_rank, seed = self.seed)
            self.train_loader = SimCLRWrapper(transform = train)
            self.val_loader = SimCLRWrapper(transform = val)
            
        elif stage == Options.inference.name:
            self.test_dataloader = SimCLRWrapper(transform = self.transform(self.DATA_PATH, batch_size = self.batch_size, input_height = self.image_size, copies = 1, stage = 'inference', num_threads = 2*self.cpus, device_id = self.local_rank, seed = self.seed))
            self.inference_dataloader = self.test_dataloader
     
    def train_dataloader(self):
        return self.train_loader
  
    def val_dataloader(self):
        return self.val_loader
    
    #give user permission to add extra arguments for SIMSIAM model particularly. This cannot share the name of any parameters from train.py
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # things we need to pass into pytorch lightning simsiam model

        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")

        # training params
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument("--lars_wrapper", action="store_true", help="apple lars wrapper over optimizer used")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=1, type=int, help="number of warmup epochs")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
        
        return parser
