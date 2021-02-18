
import os
import math
import numpy as np
import shutil
from pathlib import Path
import splitfolders
from termcolor import colored

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pytorch_lightning.loggers import WandbLogge

from argparse import ArgumentParser

#Internal Package Imports
from .models import *

def load_model(parser):
    '''
    insert comment here
    '''
    
    #load checkpoint models
    supported_techniques = {
        'SIMCLR': SIMCLR.SIMCLR,
        'SIMSIAM': SIMSIAM.SIMSIAM,
        'CLASSIFIER': CLASSIFIER.CLASSIFIER,
    }

    args = parser.parse_args()
    technique = supported_techniques[args.technique]

    
    if '.ckpt' in args.model:
        try:
            technique.load_from_checkpoint(model_args)
        except:
            print('Unable to initialize checkpoint for technique. We will initialize just the encoder for your specified technique.')
            pass
    elif True:
        print('hi')

        init_model = True


    #encoder specified
    elif 'minicnn' in encoder_name:
        #special case to make minicnn output variable output embedding size depending on user arg
        output_size =  int(''.join(x for x in encoder_name if x.isdigit()))
        encoder, embedding_size = encoders.miniCNN(output_size), output_size
        init_model = False  
    elif encoder_name == 'resnet18':
        encoder, embedding_size = encoders.resnet18(pretrained=False, first_conv=True, maxpool1=True, return_all_feature_maps=False), 512
        init_model = False
    elif encoder_name == 'imagenet_resnet18':
        encoder, embedding_size = encoders.resnet18(pretrained=True, first_conv=True, maxpool1=True, return_all_feature_maps=False), 512
        init_model = False
    elif encoder_name == 'resnet50':
        encoder, embedding_size = encoders.resnet50(pretrained=False, first_conv=True, maxpool1=True, return_all_feature_maps=False), 2048
        init_model = False
    elif encoder_name == 'imagenet_resnet50':
        encoder, embedding_size = encoders.resnet50(pretrained=True, first_conv=True, maxpool1=True, return_all_feature_maps=False), 2048
        init_model = False
    
    #try loading just the encoder
    else:
        print('Trying to initialize just the encoder from a pytorch model file (.pt)')
        try:
          model = torch.load(encoder_name)
        except:
          raise Exception('Encoder could not be loaded from path')
        try:
          embedding_size = model.embedding_size
        except:
          raise Exception('Your model specified needs to tell me its embedding size. I cannot infer output size yet. Do this by specifying a model.embedding_size in your model instance')
        init_model = False
        
    print(colored('LOAD ENCODER: ', 'blue'), encoder_name)
    return model


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--args.DATA_PATH", type=str, help="path to folders with images")
    parser.add_argument("--model", type=str, help="model to initialize. Can accept model checkpoint or just encoder name from models.py")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for SSL")
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers to use to fetch data. Typically 2 * cpus available")
    parser.add_argument("--hidden_dims", default=128, type=int, help="hidden dimensions in classification layer added onto model for finetuning")
    parser.add_argument("--epochs", default=400, type=int, help="number of epochs to train model")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for encoder")
    parser.add_argument("--linear_lr", default=1e-1, type=float, help="learning rate for classification head. Ignored when classifier technique is not called")
    parser.add_argument("--patience", default=-1, type=int, help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.")
    parser.add_argument("--val_split", default=0.2, type=float, help="percent in validation data")
    parser.add_argument("--withhold_split", default=0, type=float, help="decimal from 0-1 representing how much of the training data to withold from either training or validation. Used for experimenting with labels neeeded")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")
    parser.add_argument("--log_name", type=str, default=None, help="name of model to log on wandb and locally")
    parser.add_argument("--online_eval", default=False, type=bool, help="Finetune as a sanity check with an SSL checkpoint when testing SSL. Will be ignored if classifier technique is selected.")
    parser.add_argument("--image_size", default=256, type=int, help="height of square image")
    parser.add_argument("--resize", default=False, type=bool, help="Pre-Resize data to right shape to reduce cuda memory requirements of reading large images")
    parser.add_argument("--technique", default=None, type=str, help="SIMCLR, SIMSIAM or CLASSIFIER")
    parser.add_argument("--seed", default=1729, type=int, help="random seed for run for reproducibility")

    #add ability to parse unknown args
    args = parser.parse_args()
    
    #logging
    wandb_logger = None
    log_name = args.technique + '_' + args.log_name + '.ckpt'
    if log_name is not None:
        wandb_logger = WandbLogger(name=log_name,project='Curator')

    #resize images here
    if args.resize:
        #implement resize and modify args.DATA_PATH accordingly
        pass
    
    #Splitting Data into train and validation
    if not (os.path.isdir(f"{args.DATA_PATH}/train") and os.path.isdir(f"{args.DATA_PATH}/val")) and args.val_split != 0: 
        print(colored(f'Automatically splitting data into train and validation data...', 'blue'))
        shutil.rmtree(f'./split_data_{log_name[:-5]}', ignore_errors=True)
        splitfolders.ratio(args.DATA_PATH, output=f'./split_data_{log_name[:-5]}', ratio=(1-args.val_split-args.withhold_split, args.val_split, args.withhold_split), seed = args.seed)
        args.DATA_PATH = f'./split_data_{log_name[:-5]}'
  
    #loading model
    model = load_model(parser)

    online_evaluator = SSLOnlineEvaluator(
      drop_p=0.,
      hidden_dim=None,
      z_dim=model.embedding_size,
      num_classes=model.num_classes,
      dataset='None'
    )
    
    cbs = []
    backend = 'ddp'
    
    if args.patience > 0:
        cb = EarlyStopping('val_loss', patience = args.patience)
        cbs.append(cb)
    
    if args.online_eval and args.technique.lower() is not 'classifier':
        cbs.append(online_evaluator)
        
    trainer = pl.Trainer(gpus=args.gpus, max_epochs = args.epochs, progress_bar_refresh_rate=20, callbacks = cbs, distributed_backend=f'{backend}' if args.gpus > 1 else None, sync_batchnorm=True if args.gpus > 1 else False, logger = wandb_logger, enable_pl_optimizer = True)
    trainer.fit(model)

    Path(f"./models/").mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(f"./models/{log_name}")
    print(colored("YOUR MODEL CAN BE ACCESSED AT: ", 'blue'), f"./models/{log_name}")

if __name__ == '__main__':
    cli_main()