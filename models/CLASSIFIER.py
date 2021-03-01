import os
from termcolor import colored
import numpy as np
import math
from argparse import ArgumentParser
from termcolor import colored

import torch
from torch.nn import functional as F
from torch import nn
from torchvision.datasets import ImageFolder
from torch.optim import SGD

import pytorch_lightning as pl
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner

#Internal Imports
from dali_utils.dali_transforms import SimCLRTransform #same transform as SimCLR, but only 1 copy
from dali_utils.lightning_compat import ClassifierWrapper

class CLASSIFIER(pl.LightningModule): #SSLFineTuner

    def __init__(self, encoder, DATA_PATH, VAL_PATH, hidden_dim, image_size, seed, cpus, transform = SimCLRTransform, **classifier_hparams):
        
        data_temp = ImageFolder(DATA_PATH)
        
        self.DATA_PATH = DATA_PATH
        self.VAL_PATH = VAL_PATH
        self.transform = transform
        self.image_size = image_size
        self.cpus = cpus
        self.seed = seed
        self.num_classes = len(data_temp.classes)
        self.batch_size = classifier_hparams['batch_size']
        self.classifier_hparams = classifier_hparams
        
        self.linear_layer = SSLEvaluator(
            n_input=encoder.embedding_size,
            n_classes=self.num_classes,
            p=0.1,
            n_hidden=hidden_dim
        )
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        
#         super().__init__(backbone = encoder, 
#                          in_features = encoder.embedding_size, 
#                          num_classes = self.num_classes, 
#                          epochs = classifier_hparams['epochs'],
#                          hidden_dim = hidden_dim,
#                          dropout = classifier_hparams['dropout'],
#                          learning_rate = classifier_hparams['learning_rate'],
#                          nesterov = classifier_hparams['nesterov'],
#                          scheduler_type = classifier_hparams['scheduler_type'],
#                          decay_epochs = classifier_hparams['decay_epochs'],
#                          gamma = classifier_hparams['gamma'],
#                          final_lr = classifier_hparams['final_lr']  
#                         )
        self.encoder = encoder
        
        self.save_hyperparameters()
        print('saved hparams here')
        print(self.hparams)
  
    #override optimizer to allow modification of encoder learning rate
#     def configure_optimizers(self):
#         optimizer = SGD([
#                   {'params': self.encoder.parameters()},
#                   {'params': self.linear_layer.parameters(), 'lr': self.classifier_hparams['linear_lr']}
#               ], lr=self.classifier_hparams['learning_rate'], momentum=self.classifier_hparams['momentum'])
              
#         if self.scheduler_type == "step":
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
#         elif self.scheduler_type == "cosine":
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer,
#                 self.epochs,
#                 eta_min=self.final_lr  # total epochs to run
#             )

#         return [optimizer], [scheduler]
    
    def shared_step(self, batch):
        x, y = batch
        feats = self.encoder(x)[-1]
        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits, y)
        self.log('tloss', loss, prog_bar=True)
        self.log('tastep', acc, prog_bar=True)
        self.log('ta_epoch', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, logits, y = self.shared_step(batch)
            acc = self.val_acc(logits, y)

        acc = self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc_epoch', self.val_acc, prog_bar=True)
        self.log('val_acc_epoch', self.val_acc, prog_bar=True)
        return loss

    def loss_fn(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def configure_optimizers(self):
        opt = SGD([
                {'params': self.encoder.parameters()},
                {'params': self.linear_layer.parameters(), 'lr': 0.1}
            ], lr=1e-4, momentum=0.9)

        return [opt]


    def setup(self, stage = 'inference'):
        if stage == 'fit':
            train = self.transform(self.DATA_PATH, batch_size = self.batch_size, input_height = self.image_size, copies = 1, stage = 'train', num_threads = self.cpus, device_id = self.local_rank, seed = self.seed)
            val = self.transform(self.VAL_PATH, batch_size = self.batch_size, input_height = self.image_size, copies = 1, stage = 'validation', num_threads = self.cpus, device_id = self.local_rank, seed = self.seed)
            self.train_loader = ClassifierWrapper(transform = train)
            self.val_loader = ClassifierWrapper(transform = val)

        elif stage == 'inference':
            self.test_dataloader = ClassifierWrapper(transform = self.transform(self.DATA_PATH, batch_size = self.batch_size, input_height = self.image_size, copies = 1, stage = 'inference', num_threads = 2*self.cpus, device_id = self.local_rank, seed = self.seed))
            self.inference_dataloader = self.test_dataloader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    #give user permission to add extra arguments for SIMSIAM model particularly. This cannot share the name of any parameters from train.py
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training params
        parser.add_argument("--linear_lr", default=1e-1, type=float, help="learning rate for classification head.")
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout of neurons during training [0-1].")
        parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov during training.")
        parser.add_argument("--scheduler_type", default='cosine', type=str, help="learning rate scheduler: ['cosine' or 'step']")
        parser.add_argument("--gamma", default=0.1, type=float, help="gamma param for learning rate.")
        parser.add_argument("--decay_epochs", default=[60, 80], type=list, help="epochs to do optimizer decay")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
        parser.add_argument("--momentum", type=float, default=0.9, help="momentum for learning rate")
        return parser
