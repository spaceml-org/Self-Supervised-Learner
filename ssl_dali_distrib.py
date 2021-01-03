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

from pytorch_lightning.loggers import WandbLogger

from typing import List, Optional
from pytorch_lightning.metrics import Accuracy
from pathlib import Path
from argparse import ArgumentParser

from sklearn.metrics import f1_score, accuracy_score

from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

#internal imports
from transforms_dali import SimCLRTrainDataTransform, SimCLRValDataTransform 
from encoders_dali import load_encoder

class SIMCLR(SimCLR):

  def __init__(self, encoder, embedding_size, epochs, gpus, DATA_PATH, withhold, batch_size, val_split, hidden_dims, train_transform, val_transform, num_workers, lr, image_size):
      #data stuff

      self.DATA_PATH = DATA_PATH
      self.val_split = val_split
      self.batch_size = batch_size
      self.hidden_dims = hidden_dims
      self.train_transform = train_transform
      self.val_transform = val_transform
      self.withhold = withhold
      self.epochs = epochs
      self.num_workers = num_workers
      self.gpus = gpus
      self.lr = lr
      self.embedding_size = embedding_size
      self.image_size = image_size
      
      super().__init__(gpus = self.gpus, num_samples = 0, batch_size = self.batch_size, dataset = 'None', max_epochs = self.epochs)
      self.encoder = encoder
      
      class Projection(nn.Module):
          def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
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
            
      self.projection = Projection(input_dim = self.embedding_size, hidden_dim = self.hidden_dims)
      
      self.save_hyperparameters()
      
  def setup(self, stage = 'train'):
      #used for setting up dali pipeline, run on every gpu
      if stage == 'inference':
          print('Running model in inference mode. Dali iterator will flow data, no labels')     
          num_samples = sum([len(files) for r, d, files in os.walk(f'{self.DATA_PATH}')])
          #each gpu gets its own DALI loader
          inference_pipeline = self.val_transform(DATA_PATH = f"{self.DATA_PATH}", input_height = self.image_size, batch_size = self.batch_size, num_threads = self.num_workers, device_id = self.global_rank, stage = stage)
          
          class LightningWrapper(DALIGenericIterator):
              def __init__(self, *kargs, **kvargs):
                  super().__init__(*kargs, **kvargs)

              def __next__(self):
                  out = super().__next__()
                  out = out[0]
                  return out[self.output_map[0]]

              def __len__(self):
                return num_samples//self.batch_size

          inference_labels = [f'im{i}' for i in range(1, train_pipeline.COPIES+1)]
          print(inference_labels)
          self.inference_loader = LightningWrapper(inference_pipeline, inference_labels, auto_reset=True, fill_last_batch=False)
          self.train_loader = None
          self.val_loader = None
          
      else:
          
          num_samples = sum([len(files) for r, d, files in os.walk(f'{self.DATA_PATH}/train')])
          #each gpu gets its own DALI loader
          train_pipeline = self.train_transform(DATA_PATH = f"{self.DATA_PATH}/train", input_height = self.image_size, batch_size = self.batch_size, num_threads = self.num_workers, device_id = self.global_rank)
          print(f"{self.DATA_PATH}/train")
          val_pipeline = self.val_transform(DATA_PATH = f"{self.DATA_PATH}/val", input_height = self.image_size, batch_size = self.batch_size, num_threads = self.num_workers, device_id = self.global_rank)


          class LightningWrapper(DALIGenericIterator):
              def __init__(self, *kargs, **kvargs):
                  super().__init__(*kargs, **kvargs)

              def __next__(self):
                  out = super().__next__()
                  out = out[0]
                  return out[self.output_map[0]], torch.squeeze(out[self.output_map[-1]])

              def __len__(self):
                return num_samples//self.batch_size


          train_labels = [f'im{i}' for i in range(1, train_pipeline.COPIES+1)]
          train_labels.append('label')

          val_labels = [f'im{i}' for i in range(1, val_pipeline.COPIES+1)]
          val_labels.append('label')

          size_train = sum([len(files) for r, d, files in os.walk(f'{self.DATA_PATH}/train')])
          self.train_loader = LightningWrapper(train_pipeline, train_labels, auto_reset=True, fill_last_batch=False)
          self.val_loader = LightningWrapper(val_pipeline, val_labels, auto_reset=True, fill_last_batch=False)
          
      
  def train_dataloader(self):
      return self.train_loader
  
  def val_dataloader(self):
      return self.val_loader
  
  def inference_dataloader(self):
      return self.inference_loader
    
def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--DATA_PATH", type=str, help="path to folders with images")
    parser.add_argument("--encoder", default=None , type=str, help="encoder to initialize. Can accept SimCLR model checkpoint or just encoder name in from encoders_dali")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for SSL")
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers to use to fetch data")
    parser.add_argument("--hidden_dims", default=128, type=int, help="hidden dimensions in classification layer added onto model for finetuning")
    parser.add_argument("--epochs", default=400, type=int, help="number of epochs to train model")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for training model")
    parser.add_argument("--patience", default=-1, type=int, help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.")
    parser.add_argument("--val_split", default=0.2, type=float, help="percent in validation data")
    parser.add_argument("--withhold_split", default=0, type=float, help="decimal from 0-1 representing how much of the training data to withold from either training or validation. Used for experimenting with labels neeeded")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")
    parser.add_argument("--log_name", type=str, help="name of model to log on wandb and locally")
    parser.add_argument("--online_eval", default=False, type=bool, help="Do finetuning on model if labels are provided as a sanity check")
    parser.add_argument("--image_size", default=256, type=int, help="height of square image")
    
    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH
    batch_size = args.batch_size
    num_workers = args.num_workers
    hidden_dims = args.hidden_dims
    epochs = args.epochs
    lr = args.lr
    patience = args.patience
    val_split = args.val_split
    withhold = args.withhold_split
    gpus = args.gpus
    encoder = args.encoder
    log_name = 'SIMCLR_SSL_' + args.log_name + '.ckpt'
    online_eval = args.online_eval
    image_size = args.image_size
    
    wandb_logger = WandbLogger(name=log_name,project='SpaceForce')
    checkpointed = '.ckpt' in encoder    
    
    if not (path.isdir(f"{DATA_PATH}/train") and path.isdir(f"{DATA_PATH}/val")): 
        shutil.rmtree(f'./split_data_{log_name[:-5]}', ignore_errors=True)
        splitfolders.ratio(DATA_PATH, output=f'./split_data_{log_name[:-5]}', ratio=(1-val_split-withhold, val_split, withhold), seed = 10)
        DATA_PATH = f'./split_data_{log_name[:-5]}'
        print(f'automatically splitting data into train and validation data {val_split} and withhold {withhold}')

    num_classes = len(os.listdir(f'{DATA_PATH}/train'))
        
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
        model = SIMCLR(encoder = encoder, embedding_size = embedding_size, gpus = gpus, epochs = epochs, DATA_PATH = DATA_PATH, withhold = withhold, batch_size = batch_size, val_split = val_split, hidden_dims = hidden_dims, train_transform = SimCLRTrainDataTransform, val_transform = SimCLRValDataTransform, num_workers = num_workers, lr = lr, image_size = image_size)
        
    online_evaluator = SSLOnlineEvaluator(
      drop_p=0.,
      hidden_dim=None,
      z_dim=embedding_size,
      num_classes=num_classes,
      dataset='None'
    )
    
    cbs = []
    backend = 'ddp'
    
    if patience > 0:
        cb = EarlyStopping('val_loss', patience = patience)
        cbs.append(cb)
    
    if online_eval:
        cbs.append(online_evaluator)
        
    trainer = Trainer(gpus=gpus, max_epochs = epochs, progress_bar_refresh_rate=20, callbacks = cbs, distributed_backend=f'{backend}' if args.gpus > 1 else None, logger = wandb_logger, enable_pl_optimizer=True)
    
    trainer.fit(model)
    Path(f"./models/SSL").mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(f"./models/SSL/{log_name}")
    
if __name__ == '__main__':
    cli_main()
