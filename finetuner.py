from typing import List, Optional
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
from pl_bolts.models.self_supervised import SSLEvaluator
from pathlib import Path
from argparse import ArgumentParser
from pl_bolts.models.self_supervised.resnets import resnet18
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRFinetuneTransform
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from tqdm import tqdm

#imports from internal
from CustomDataset import FolderDataset
from SSLTrainer import Projection

class SSLFineTuner(pl.LightningModule):

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 2048,
        num_classes: int = 1000,
        epochs: int = 100,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = 'cosine',
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr

        self.backbone = backbone
        self.linear_layer = SSLEvaluator(
            n_input=in_features,
            n_classes=num_classes,
            p=dropout,
            n_hidden=hidden_dim
        )

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.backbone.train()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc_step', acc, prog_bar=True)
        self.log('train_acc_epoch', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.val_acc(logits, y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.test_acc(logits, y)

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc)

        return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            list(self.backbone.parameters()) + list(self.linear_layer.parameters()),
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.decay_epochs, gamma=self.gamma
            )
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]

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
    pretrain = args.pretrain_encoder
    withold_train_percent = args.withold_train_percent
    version = args.version
    model_checkpoint = args.MODEL_PATH
    gpus = args.gpus
    eval_model = args.eval
    version = args.version

    # #testing
    # batch_size = 128
    # image_type = 'tif'
    # image_size = 256
    # num_workers = 4
    # DATA_PATH ='/content/UCMerced_LandUse/Images'
    # embedding_size = 128
    # epochs = 15
    # hidden_dims = 128
    # lr = 1e-3
    # patience = 1
    # val_split = 0.2
    # withold_train_percent = 0.2
    # model_checkpoint = '/content/models/SSL/SIMCLR_SSL_0/SIMCLR_SSL_0.pt'
    # gpus = 1
    # eval_model = True
    # version = "0"

    #gets dataset. We can't combine since validation data has different transform needed
    finetune_dataset = FolderDataset(DATA_PATH, validation = False, 
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
    finetune_val_dataset = FolderDataset(DATA_PATH, validation = True,
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
    model.encoder = resnet18(pretrained=False, first_conv=model.first_conv, maxpool1=model.maxpool1, return_all_feature_maps=False)
    model.projection = Projection(input_dim = 512, hidden_dim = 256, output_dim = embedding_size) #overrides
    model.load_state_dict(torch.load(model_checkpoint))
    print('Successfully loaded your checkpoint. Keep in mind that this does not preserve the previous trainer states, only the model weights')

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

if __name__ == '__main__':
    cli_main()
