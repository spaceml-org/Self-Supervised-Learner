from typing import List, Optional
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
from pl_bolts.models.self_supervised import SSLEvaluator
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from tqdm import tqdm

from pl_bolts.models.self_supervised.simclr.transforms import SimCLRFinetuneTransform
from pl_bolts.models.self_supervised import SimCLR
import torch
from pl_bolts.models.self_supervised.resnets import resnet18
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.datasets import ImageFolder
from CustomDataset import FolderDataset2
from SSLTrainer2 import Projection
from ssl_finetuner import SSLFineTuner

  
import pytorch_lightning as pl
import splitfolders
from torchvision.datasets import ImageFolder
from os import path
from torch.utils.data import DataLoader
import shutil


def eval_finetune(tuner, kind, loader, save_path):
    y_preds = torch.empty(0)
    y_true = torch.empty(0)
    tuner.eval()
    with torch.no_grad():
      for batch in tqdm(loader):
        X, y = batch
        y_true = torch.cat((y_true, y))
        loss, logits, _ = tuner.shared_step((X, y))
        preds = torch.argmax(logits, dim=1)
        y_preds= torch.cat((y_preds, preds))

    cm = confusion_matrix(y_true, y_preds, normalize='true')

    v = len(cm)
    df_cm = pd.DataFrame(cm, range(v), range(v))
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}) # font size
    f1 =  f1_score(y_true, y_preds, average = 'weighted')
    print(f'F1 score - {kind}: ', f1)
    print('ACCURACY: ', accuracy_score(y_true, y_preds))
    plt.title(f'Confusion matrix for finetuned classifier - {kind}')
    plt.savefig(f'{save_path}/ConfusionMatrix.png', dpi=400)
    print('Confusion Matrix can be found here: ', f'{save_path}/ConfusionMatrix.png')
    
def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--DATA_PATH", type=str, help="path to folders with images")
    parser.add_argument("--MODEL_PATH", default=None , type=str, help="path to model checkpoint")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for SSL")
    parser.add_argument("--image_size", default=256, type=int, help="image size for SSL")
    parser.add_argument("--image_type", default="tif", type=str, help="extension of image for PIL to open and parse - i.e. jpeg, gif, tif, etc. Only put the extension name, not the dot (.)")
    parser.add_argument("--num_workers", default=1, type=int, help="number of CPU cores to use for data processing")
    parser.add_argument("--image_embedding_size", default=128, type=int, help="size of image representation of SIMCLR")
    parser.add_argument("--hidden_dims", default=128, type=int, help="hidden dimensions in classification layer added onto model for finetuning")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs to train model")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for training model")
    parser.add_argument("--patience", default=-1, type=int, help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.")
    parser.add_argument("--val_split", default=0.2, type=float, help="percent in validation data")
    parser.add_argument("--withold_train_percent", default=0, type=float, help="decimal from 0-1 representing how much of the training data to withold during finetuning")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")
    parser.add_argument("--eval", default=True, type=bool, help="Eval Mode will train and evaluate the finetuned model's performance")
    parser.add_argument("--imagenet_weights", default=False, type=bool, help="Use weights from a non-SSL")
    parser.add_argument("--version", default="0", type=str, help="version to name checkpoint for saving")
    
    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH
    batch_size = args.batch_size
    image_size = args.image_size
    image_type = args.image_type
    num_workers = args.num_workers
    embedding_size = args.image_embedding_size
    hidden_dims = args.hidden_dims
    epochs = args.epochs
    lr = args.lr
    patience = args.patience
    val_split = args.val_split
    withold_train_percent = args.withold_train_percent
    version = args.version
    model_checkpoint = args.MODEL_PATH
    gpus = args.gpus
    eval_model = args.eval
    version = args.version
    imagenet_weights = args.imagenet_weights

    #gets dataset. We can't combine since validation data has different transform needed
    finetune_dataset = FolderDataset2(DATA_PATH, validation = False, 
                                  val_split = val_split, 
                                  withold_train_percent = withold_train_percent, 
                                  transform = SimCLRFinetuneTransform(image_size), 
                                  image_type = image_type
                                  ) 
    
    finetune_loader = torch.utils.data.DataLoader(finetune_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last = True
                                              )
    

    print('Training Data Loaded...')
    finetune_val_dataset = FolderDataset2(DATA_PATH, validation = True,
                                val_split = val_split,
                                transform =SimCLRFinetuneTransform(image_size),
                                image_type = image_type
                               )
    
    finetune_val_loader = torch.utils.data.DataLoader(finetune_val_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last = True
                                            )
    print('Validation Data Loaded...')
    
    num_samples = len(finetune_dataset)
    model = SimCLR(arch = 'resnet18', batch_size = batch_size, num_samples = num_samples, gpus = gpus, dataset = 'None', max_epochs = epochs, learning_rate = lr) #
    model.encoder = resnet18(pretrained= imagenet_weights, first_conv=model.first_conv, maxpool1=model.maxpool1, return_all_feature_maps=False)
    model.projection = Projection(input_dim = 512, hidden_dim = 256, output_dim = embedding_size) #overrides
    
    if model_checkpoint is not None:  
        model.load_state_dict(torch.load(model_checkpoint))
        print('Successfully loaded your checkpoint. Keep in mind that this does not preserve the previous trainer states, only the model weights')
    else:
        if imagenet_weights:   
            print('Using imagenet weights instead of a pretrained SSL model')
        else:
            print('Using random initialization of encoder')
        
    num_classes = len(set(finetune_dataset.labels))
    print('Finetuning to classify ', num_classes, ' Classes')

    tuner = SSLFineTuner(model, in_features=512, num_classes=num_classes, hidden_dim=hidden_dims, learning_rate=lr)
    if patience > 0:
      cb = EarlyStopping('val_loss', patience = patience)
      trainer = Trainer(gpus=gpus, max_epochs = epochs, callbacks=[cb], progress_bar_refresh_rate=5)
    else:
      trainer = Trainer(gpus=gpus, max_epochs = epochs, progress_bar_refresh_rate=5)
    tuner.cuda()
    trainer.fit(tuner, train_dataloader= finetune_loader, val_dataloaders=finetune_val_loader)

    Path(f"./models/Finetune/SIMCLR_Finetune_{version}").mkdir(parents=True, exist_ok=True)
    
    if eval_model:
      print('Evaluating Model...')
      save_path = f"./models/Finetune/SIMCLR_Finetune_{version}/Evaluation/trainingMetrics"
      Path(save_path).mkdir(parents=True, exist_ok=True)
      eval_finetune(tuner, 'training', finetune_loader, save_path)

      save_path = f"./models/Finetune/SIMCLR_Finetune_{version}/Evaluation/validationMetrics"
      Path(save_path).mkdir(parents=True, exist_ok=True)
      eval_finetune(tuner, 'validation', finetune_val_loader, save_path)
    
    print('Saving model...')
    
    torch.save(tuner.state_dict(), f"./models/Finetune/SIMCLR_Finetune_{version}/SIMCLR_FINETUNE_{version}.pt")
    
    if eval_model:
      print('Evaluating Model...')
      save_path = f"./models/Finetune/SIMCLR_Finetune_{version}/Evaluation/trainingMetrics"
      Path(save_path).mkdir(parents=True, exist_ok=True)
      eval_finetune(tuner, 'training', finetune_loader, save_path)

      save_path = f"./models/Finetune/SIMCLR_Finetune_{version}/Evaluation/validationMetrics"
      Path(save_path).mkdir(parents=True, exist_ok=True)
      eval_finetune(tuner, 'validation', finetune_val_loader, save_path)
    
    print('Saving model...')
    
    torch.save(tuner.state_dict(), f"./models/Finetune/SIMCLR_Finetune_{version}/SIMCLR_FINETUNE_{version}.pt")

if __name__ == '__main__':
    cli_main()
