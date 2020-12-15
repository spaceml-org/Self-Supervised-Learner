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

from ImageDataModule import ImageDataModule
from SSLTrainer2 import Projection
from ssl_finetuner import SSLFineTuner
from CustomDataset import FolderDataset2

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
    parser.add_argument("--image_embedding_size", default=128, type=int, help="size of image representation of SIMCLR")
    parser.add_argument("--hidden_dims", default=128, type=int, help="hidden dimensions in classification layer added onto model for finetuning")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs to train model")
    parser.add_argument("--lr", default=0.3, type=float, help="learning rate for training model")
    parser.add_argument("--patience", default=-1, type=int, help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.")
    parser.add_argument("--val_split", default=0.2, type=float, help="percent in validation data")
    parser.add_argument("--withold_train_percent", default=0, type=float, help="decimal from 0-1 representing how much of the training data to withold during finetuning")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")
    parser.add_argument("--eval", default=True, type=bool, help="Eval Mode will train and evaluate the finetuned model's performance")
    parser.add_argument("--pretrain_encoder", default=False, type=bool, help="initialize resnet encoder with pretrained imagenet weights. Ignored if MODEL_PATH is specified.")
    parser.add_argument("--version", default="0", type=str, help="version to name checkpoint for saving")
    parser.add_argument("--fix_backbone", default=True, type=bool, help="Fix backbone during finetuning")
    parser.add_argument("--num_workers", default=0, type=int, help="number of workers to use to fetch data")
    
    args = parser.parse_args()
    URL = args.DATA_PATH
    batch_size = args.batch_size
    image_size = args.image_size
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
    pretrain= args.pretrain_encoder
    fix_backbone = args.fix_backbone
    num_workers = args.num_workers
    
    dm = FolderDataset2(URL, val_split = val_split, train_transform = SimCLRFinetuneTransform(image_size), val_transform = SimCLRFinetuneTransform(image_size))
    dm.setup()
    
    model = SimCLR(arch = 'resnet18', batch_size = batch_size, num_samples = dm.num_samples, gpus = 1, dataset = 'None', max_epochs = 100, learning_rate = lr) #
    model.projection = Projection(input_dim = 512, hidden_dim = 256, output_dim = 128) #overrides
    model.encoder =  resnet18(pretrained=pretrain, first_conv=model.first_conv, maxpool1=model.maxpool1, return_all_feature_maps=False)
    if model_checkpoint is not None:
        model.load_state_dict(torch.load(model_checkpoint))
        print('Successfully loaded your checkpoint. Keep in mind that this does not preserve the previous trainer states, only the model weights')
    else:
        if pretrain:   
            print('Using imagenet weights instead of a pretrained SSL model')
        else:
            print('Using random initialization of encoder')
        
    print('Finetuning to classify ', dm.num_classes, ' Classes')
    

    
    tuner = SSLFineTuner(
      model,
      in_features=512,
      num_classes=dm.num_classes,
      epochs=epochs,
      hidden_dim=hidden_dims,
      dropout=0,
      learning_rate=lr,
      weight_decay=1e-6,
      nesterov=False,
      scheduler_type='cosine',
      gamma=0.1,
      final_lr=0.,
      fix_backbone = True
    )

    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=1,
        precision=16,
        max_epochs=epochs,
        distributed_backend='ddp',
        sync_batchnorm=False
    )

    trainer.fit(tuner, dm)
    
    Path(f"./models/Finetune/SIMCLR_Finetune_{version}").mkdir(parents=True, exist_ok=True)
    
    if eval_model:  
      print('Evaluating Model...')
      save_path = f"./models/Finetune/SIMCLR_Finetune_{version}/Evaluation/validationMetrics"
      Path(save_path).mkdir(parents=True, exist_ok=True)
    
      if dm.val_dataloader() is not None:
        eval_finetune(tuner, 'validation', dm.val_dataloader(), save_path)
        
      save_path = f"./models/Finetune/SIMCLR_Finetune_{version}/Evaluation/trainingMetrics"
      Path(save_path).mkdir(parents=True, exist_ok=True)
      eval_finetune(tuner, 'training', dm.train_dataloader(), save_path)
    
    print('Saving model...')
    
    torch.save(tuner.state_dict(), f"./models/Finetune/SIMCLR_Finetune_{version}/SIMCLR_FINETUNE_{version}.pt")

if __name__ == '__main__':
    cli_main()
