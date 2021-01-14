from termcolor import colored
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
import pytorch_lightning as pl
import shutil
import os
from os import path
import splitfolders
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import math
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from typing import List, Optional
from pytorch_lightning.metrics import Accuracy
from pathlib import Path
from argparse import ArgumentParser

from sklearn.metrics import f1_score, accuracy_score
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

from ssl_dali_distrib import SIMCLR
from transforms_dali import SimCLRTrainDataTransform
from encoders_dali import load_encoder


def cli_main(size, DATA_PATH, batch_size, num_workers, hidden_dims, epochs, lr, 
          patience, val_split, withhold, gpus, encoder, log_name, online_eval):
    
    wandb_logger = WandbLogger(name=log_name,project='SpaceForce')
    checkpointed = '.ckpt' in encoder    
    print("DATA_PATH",DATA_PATH)
    if not (path.isdir(f"{DATA_PATH}/train") and path.isdir(f"{DATA_PATH}/val")): 
        print(colored(f'Automatically splitting data into train and validation data...', 'blue'))
        shutil.rmtree(f'./split_data_{log_name[:-5]}', ignore_errors=True)
        splitfolders.ratio(DATA_PATH, output=f'./split_data_{log_name[:-5]}', ratio=(1-val_split-withhold, val_split, withhold), seed = 10)
        DATA_PATH = f'./split_data_{log_name[:-5]}'

    if checkpointed:
        print('Resuming SSL Training from Model Checkpoint')
        try:
            model = SIMCLR.load_from_checkpoint(checkpoint_path=encoder)
            embedding_size = model.embedding_size
        except Exception as e:
            print(e)
            print('invalid checkpoint to initialize SIMCLR. This checkpoint needs to include the encoder and projection and is of the SIMCLR class from this library. Will try to initialize just the encoder')
            checkpointed = False         
    elif not checkpointed:
        encoder, embedding_size = load_encoder(encoder)
        model = SIMCLR(image_size = size, encoder = encoder, embedding_size = embedding_size, gpus = gpus, epochs = epochs, DATA_PATH = DATA_PATH, withhold = withhold, batch_size = batch_size, val_split = val_split, hidden_dims = hidden_dims, train_transform = SimCLRTrainDataTransform, val_transform = SimCLRTrainDataTransform, num_workers = num_workers, lr = lr)
    print("DATA_PATH after Not checkpointed",DATA_PATH)

    cbs = []
    backend = 'dp'
    
    if patience > 0:
        cb = EarlyStopping('val_loss', patience = patience)
        cbs.append(cb)
    
    if online_eval:
        num_classes = len(os.listdir(f'{DATA_PATH}/train'))   
        online_evaluator = SSLOnlineEvaluator(
          drop_p=0.,
          hidden_dim=None,
          z_dim=embedding_size,
          num_classes=num_classes,
          dataset='None'
        )
        cbs.append(online_evaluator)
        backend = 'ddp'
        
    trainer = Trainer(gpus=gpus, max_epochs = epochs, progress_bar_refresh_rate=5, callbacks = cbs, distributed_backend=f'{backend}' if gpus > 1 else None, logger = wandb_logger, enable_pl_optimizer=True)
    
    print('USING BACKEND______________________________ ', backend)
    trainer.fit(model)
    Path(f"./models/SSL").mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(f"./models/SSL/{log_name}")
    #torch.save(model.encoder.state_dict(), f"./models/SSL/SIMCLR_SSL_{version}/SIMCLR_SSL_{version}.pt")
    return f"./models/SSL/{log_name}"
if __name__ == '__main__':
    cli_main()