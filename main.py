import torch
from torch.nn import functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
import torchvision.datasets as datasets
import pytorch_lightning as pl
import shutil
import os
from imutils import paths 
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
from transforms_dali import SimCLRTrainDataTransform
from encoders_dali import load_encoder
from ssl_dali_distrib import cli_main, SIMCLR
from utils import plot_metrics, plot_umap, get_embeddings, n_random_subset, prepare_dataset, class_distrib, farthest_point, min_max_diverse_embeddings, animate_umap
from cli_main import cli_main
from TSNE import TSNE_visualiser

def driver():

  parser = ArgumentParser()
  parser.add_argument("--image_size", default = 256, type=int, help="Size of the image")
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
  parser.add_argument("--num_iterations", default=1, type=int, help="Number of times pretraining occurs")
  parser.add_argument("--subset_size", default= 0.1, type = int, help= "size of the subset of dataset that goes into SimCLR training")
  parser.add_argument("--buffer_dataset_path", type= str, help = "Where the subsets are stored everytime before passing into SimCLR")
  parser.add_argument("--metric", default="count", type=str, help="Type of Metric to evaluate pretraining")

  args = parser.parse_args()
  size = args.image_size
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
  num_iters = args.num_iterations
  subset_size = args.subset_size
  buffer_dataset_path = args.buffer_dataset_path
  metric = args.metric

  train_image_paths = list(paths.list_images(DATA_PATH + '/train'))
  random_points_fnames = n_random_subset(subset_size, train_image_paths)
  prepare_dataset(buffer_dataset_path , random_points_fnames)

  transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([size, size], interpolation=2)])
  val_data = datasets.ImageFolder(root=DATA_PATH+'/train', transform=transform) 
  dataset_paths = [pair[0] for pair in val_data.samples]
  val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=False)
  metric_array = []

  #creating a folder for storing graphs
  try:
    os.mkdir("./graphs/")
  except:
    shutil.rmtree("./graphs/")
    os.mkdir("./graphs/")
    os.mkdir("./graphs/DiversityAlgorithm/")
    os.mkdir("./graphs/Full_Dataset/")
    os.mkdir("./graphs/TSNE/")
  for i in range(num_iters):
    print("-----------------Iteration: ",i+1,"----------------------")
    ckpt = cli_main(size, buffer_dataset_path+"/train", batch_size, num_workers, hidden_dims, epochs, lr, 
            patience, val_split, withhold, gpus, encoder, log_name, online_eval)
    encoder = ckpt
    print("Obtaining Embeddings")
    embedding = get_embeddings(encoder, val_loader)
    metric_array.append(class_distrib(buffer_dataset_path, metric= metric))
    print("Applying Diversity Algorithm...")
    da_files, da_embeddings, da_distances = min_max_diverse_embeddings(subset_size, dataset_paths,  embedding, i = farthest_point(embedding))
    print("Preparing New Dataset...")
    prepare_dataset(buffer_dataset_path, da_files)
    print("New Dataset abiding formats successfully created")
    print("Number and Shape of Embeddings:", len(embedding),embedding[0].shape)
    plot_umap(da_embeddings, da_files, count= i, path="./graphs/DiversityAlgorithm/")
    plot_umap(embedding, dataset_paths, count= i, path="./graphs/Full_Dataset/")
    ##TSNE 
    print("Starting TSNE")
    da_tsne = TSNE_visualiser(da_embeddings, da_files)
    print("KNN Clusters created")
    neighbors, distances, indices = da_tsne.knn_cluster(da_tsne.feature_list)
    print("KNN Clusters created, onto TSNE")
    tsne_results = da_tsne.fit_tsne(da_tsne.feature_list)
    print("TSNE Fit complete")
    da_tsne.scatter_plot(tsne_results, da_tsne.labels, "./graphs/TSNE/", i)
    da_tsne.show_tsne(tsne_results[:, 0], tsne_results[:, 1], da_tsne.filenames, "./graphs/TSNE/", i)
    da_tsne.tsne_to_grid_plotter_manual(tsne_results[:, 0], tsne_results[:, 1], da_tsne.filenames, "./graphs/TSNE/", i)
    print("Graphs created and stored")
  imate_umap("./graphs/DiversityAlgorithm", fps = 1)
  animate_umap("./graphs/Full_Dataset",fps=1)
  if metric != "count":
    print(metric_array)
    plot_metrics(metric, metric_array)



if __name__ == "__main__":
  driver()
  
