import string
import cv2
import random
from PIL import Image
import pl_bolts
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning import Trainer
from pl_bolts.models.self_supervised.evaluator import Flatten
import os
import numpy as np
import pandas as pd
import torch
import gc
from tqdm.notebook import tqdm 
import time
import random
import scann
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import h5py
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform, SimCLRFinetuneTransform
from argparse import ArgumentParser
from pl_bolts.models.self_supervised.resnets import resnet18
import torch.nn as nn
import torch.nn.functional as F

#imports from internal
from CustomDataset import FolderDataset

from pathlib import Path

#this is from their code
class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lin = nn.Linear(self.input_dim, self.hidden_dim)
        self.b = nn.BatchNorm1d(self.hidden_dim)
        self.l =nn.Linear(self.hidden_dim, self.output_dim, bias=False)

    def forward(self, x):
        x = self.lin(x)
        x = F.relu(self.b(x))
        x = self.l(x)
        return F.normalize(x, dim=1)


def cli_main():
    
    parser = ArgumentParser()
    parser.add_argument("--DATA_PATH", type=str, help="path to folders with images")
    parser.add_argument("--MODEL_PATH", default=None , type=str, help="path to model checkpoint")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for SSL")
    parser.add_argument("--image_size", default=256, type=int, help="image size for SSL")
    parser.add_argument("--image_type", default="tif", type=str, help="extension of image for PIL to open and parse - i.e. jpeg, gif, tif, etc. Only put the extension name, not the dot (.)")
    parser.add_argument("--num_workers", default=1, type=int, help="number of CPU cores to use for data processing")
    parser.add_argument("--image_embedding_size", default=128, type=int, help="size of image representation of SIMCLR")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs to train model")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for training model")
    parser.add_argument("--patience", default=-1, type=int, help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.")
    parser.add_argument("--val_split", default=0.2, type=float, help="percent in validation data")
    parser.add_argument("--pretrain_encoder", default=False, type=bool, help="initialize resnet encoder with pretrained imagenet weights. Cannot be true if passing previous SSL model checkpoint.")
    parser.add_argument("--withold_train_percent", default=0, type=float, help="decimal from 0-1 representing how much of the training data to withold during SSL training")
    parser.add_argument("--version", default="0", type=str, help="version to name checkpoint for saving")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")

    args = parser.parse_args()
    URL = args.DATA_PATH
    batch_size = args.batch_size
    image_size = args.image_size
    image_type = args.image_type
    num_workers = args.num_workers
    embedding_size = args.image_embedding_size
    epochs = args.epochs
    lr = args.lr
    patience = args.patience
    val_split = args.val_split
    pretrain = args.pretrain_encoder
    withold_train_percent = args.withold_train_percent
    version = args.version
    model_checkpoint = args.MODEL_PATH
    gpus = args.gpus

    # #testing
    # batch_size = 128
    # image_type = 'tif'
    # image_size = 256
    # num_workers = 4
    # URL ='/content/UCMerced_LandUse/Images'
    # embedding_size = 128
    # epochs = 2
    # lr = 1e-3
    # patience = 1
    # val_split = 0.2
    # pretrain = False
    # withold_train_percent = 0.2
    # version = "1"
    # model_checkpoint = '/content/models/SSL/SIMCLR_SSL_0.pt'
    # gpus = 1


    # #gets dataset. We can't combine since validation data has different transform needed
    train_dataset = FolderDataset(URL, validation = False, 
                                  val_split = val_split, 
                                  withold_train_percent = withold_train_percent, 
                                  transform = SimCLRTrainDataTransform(image_size), 
                                  image_type = image_type
                                  ) 
    
    data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last = True
                                              )
    

    print('Training Data Loaded...')
    val_dataset = FolderDataset(URL, validation = True,
                                val_split = val_split,
                                transform = SimCLREvalDataTransform(image_size))
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last = True
                                            )
    print('Validation Data Loaded...')
    
    num_samples = len(train_dataset)


    #init model with batch size, num_samples (len of data), epochs to train, and autofinds learning rate
    model = SimCLR(arch = 'resnet18', batch_size = batch_size, num_samples = num_samples, gpus = gpus, dataset = 'None', max_epochs = epochs, learning_rate = lr) #
    
    model.encoder = resnet18(pretrained=pretrain, first_conv=model.first_conv, maxpool1=model.maxpool1, return_all_feature_maps=False)
    model.projection = Projection(input_dim = 512, hidden_dim = 256, output_dim = embedding_size) #overrides
       
    if patience > 0:
      cb = EarlyStopping('val_loss', patience = patience)
      trainer = Trainer(gpus=gpus, max_epochs = epochs, callbacks=[cb], progress_bar_refresh_rate=5)
    else:
      trainer = Trainer(gpus=gpus, max_epochs = epochs, progress_bar_refresh_rate=5)

    if model_checkpoint is not None:
      model.load_state_dict(torch.load(model_checkpoint))
      print('Successfully loaded your checkpoint. Keep in mind that this does not preserve the previous trainer states, only the model weights')

    model.cuda()

    print('Model Initialized')
    trainer.fit(model, data_loader, val_loader)
    
    Path("./models/SSL/").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"./models/SSL/SIMCLR_SSL_{version}.pt")

    

if __name__ == '__main__':
    cli_main()


