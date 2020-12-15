#internal imports
from pytorch_lightning.loggers import CometLogger
from transforms import SimCLRFinetuneTrainDataTransform
from encoders import load_encoder

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

import pytorch_lightning as pl
import shutil
import os
from os import path
import splitfolders
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from typing import List, Optional
from pytorch_lightning.metrics import Accuracy
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator




import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam

class finetuneSIMCLR(pl.LightningModule):

  def __init__(self, encoder, DATA_PATH, batch_size, val_split, hidden_dims, train_transform, val_transform, num_workers, **kwargs):
      super().__init__()

      self.DATA_PATH = DATA_PATH
      self.val_split = val_split
      self.batch_size = batch_size
      self.hidden_dims = hidden_dims
      self.train_transform = train_transform
      self.val_transform = val_transform
      self.num_workers = num_workers
      
      #data stuff
      shutil.rmtree('split_data', ignore_errors=True)
      if not (path.isdir(f"{self.DATA_PATH}/train") and path.isdir(f"{self.DATA_PATH}/val")): 
          splitfolders.ratio(self.DATA_PATH, output=f"split_data", ratio=(1-self.val_split, self.val_split), seed = 10)
          self.DATA_PATH = 'split_data'
          print('automatically splitting data into train and validation data')

      self.num_classes = len(os.listdir(f'{self.DATA_PATH}/train'))

      #model stuff    
      self.eval_acc = Accuracy()
      self.encoder, self.embedding_size = load_encoder(encoder, kwargs)
      self.fc1 = nn.Linear(self.embedding_size, self.hidden_dims)
      self.fc2 = nn.Linear(self.hidden_dims, self.num_classes)


  def process_batch(self, batch):
      return batch

  def forward(self, x):
      x = self.encoder(x)[0]
      x = F.log_softmax(self.fc1(x), dim = 1)
      return x

  def shared_step(self, batch):
      x, y = self.process_batch(batch)
      logits = self(x)
      loss = self.loss_fn(logits, y)
      return loss, logits, y

  def training_step(self, batch, batch_idx):
      loss, logits, y = self.shared_step(batch)
      acc = self.eval_acc(logits, y)
      self.log('train_loss', loss, prog_bar=True)
      #self.log('train_acc_step', acc)
      #self.log('train_acc_epoch', self.eval_acc, prog_bar=True)

      return loss

  def validation_step(self, batch, batch_idx):
      with torch.no_grad():
          loss, logits, y = self.shared_step(batch)
          acc = self.eval_acc(logits, y)
      #self.log('val_loss', loss, prog_bar=True)
      #self.log('val_acc_step', acc)
      #self.log('val_acc_epoch', self.eval_acc, prog_bar=True)

      return loss

  def loss_fn(self, logits, labels):
      return F.cross_entropy(logits, labels)

  def configure_optimizers(self):
      params = list(self.encoder.parameters()) + list(self.parameters())
      return Adam(params, lr=1e-3)
  
  
  def prepare_data(self):

      train_pipeline = self.train_transform(DATA_PATH = f"{self.DATA_PATH}/train", input_height = 256, batch_size = self.batch_size, num_threads = self.num_workers, device_id = 0)
      print(f"{self.DATA_PATH}/train")
      val_pipeline = self.val_transform(DATA_PATH = f"{self.DATA_PATH}/val", input_height = 256, batch_size = self.batch_size, num_threads = self.num_workers, device_id = 0)

      class LightningWrapper(DALIClassificationIterator):
          def __init__(self, *kargs, **kvargs):
              super().__init__(*kargs, **kvargs)

          def __next__(self):
              out = super().__next__()
              out = out[0]
              return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

      self.train_loader = LightningWrapper(train_pipeline, fill_last_batch=False, auto_reset=True, reader_name = "Reader")
      self.val_loader = LightningWrapper(val_pipeline, fill_last_batch=False, auto_reset=True, reader_name = "Reader")


  def train_dataloader(self):
       return self.train_loader
  
  def val_dataloader(self):
       return self.val_loader
       
def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--DATA_PATH", type=str, help="path to folders with images")
    parser.add_argument("--MODEL_PATH", default=None, type=str, help="path to model checkpoint.")
    parser.add_argument("--encoder", default=None , type=str, help="encoder for model found in encoders.py")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for SSL")
    parser.add_argument("--num_workers", default=0, type=int, help="number of workers to use to fetch data")
    parser.add_argument("--hidden_dims", default=128, type=int, help="hidden dimensions in classification layer added onto model for finetuning")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs to train model")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for training model")
    parser.add_argument("--patience", default=-1, type=int, help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.")
    parser.add_argument("--val_split", default=0.2, type=float, help="percent in validation data")
    parser.add_argument("--withold_train_percent", default=0, type=float, help="decimal from 0-1 representing how much of the training data to withold during finetuning")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")
    parser.add_argument("--eval", default=True, type=bool, help="Eval Mode will train and evaluate the finetuned model's performance")
    parser.add_argument("--pretrain_encoder", default=False, type=bool, help="initialize resnet encoder with pretrained imagenet weights. Ignored if MODEL_PATH is specified.")
    parser.add_argument("--version", default="0", type=str, help="version to name checkpoint for saving")
    
    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH
    batch_size = args.batch_size
    num_workers = args.num_workers
    hidden_dims = args.hidden_dims
    epochs = args.epochs
    lr = args.lr
    patience = args.patience
    val_split = args.val_split
    withold_train_percent = args.withold_train_percent
    version = args.version
    MODEL_PATH = args.MODEL_PATH
    gpus = args.gpus
    eval_model = args.eval
    version = args.version
    pretrain = args.pretrain_encoder
    encoder = args.encoder
    
    
#     model = finetuneSIMCLR(encoder = encoder, pretrained = pretrain, DATA_PATH  = DATA_PATH, batch_size = batch_size, val_split = val_split, hidden_dims = hidden_dims, train_transform = SimCLRFinetuneTrainDataTransform, val_transform = SimCLRFinetuneTrainDataTransform, num_workers = num_workers)
#     if patience > 0:
#         cb = EarlyStopping('val_loss', patience = patience)
#         trainer = Trainer(gpus=gpus, max_epochs = epochs, callbacks=[cb], progress_bar_refresh_rate=5)
#     else:
#         trainer = Trainer(gpus=gpus, max_epochs = epochs, progress_bar_refresh_rate=5)

#     trainer.fit(model)
    comet_logger = CometLogger(save_dir='logs/')
    print('INIT LOGGER HERE ABOVE__________________________')
    model = finetuneSIMCLR(encoder = 'resnet18', pretrained = True, DATA_PATH  = '/content/UCMerced_LandUse/Images', batch_size = 64, val_split = 0.2, hidden_dims = 128, train_transform = SimCLRFinetuneTrainDataTransform, val_transform = SimCLRFinetuneTrainDataTransform, num_workers = 4)
    trainer = Trainer(gpus=1, distributed_backend="ddp", max_epochs=5, logger=comet_logger )
    trainer.fit(model)
    
if __name__ == '__main__':
    cli_main()
