from argparse import ArgumentParser
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.resnets import resnet18
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pathlib import Path
import torch
import os
import scann
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

#imports from internal
from CustomDataset import FolderDataset
from SSLTrainer import Projection


def eval_embeddings(model, dataset, save_path):
  model.eval()
  embeddings_matrix = torch.empty((0, 512)).cuda()
  for batch in tqdm(dataset):
      #this mirrors shared_step
      (image, im1, _), y = batch
      with torch.no_grad():
        image = torch.unsqueeze(image, 0)
        image = image.cuda()
        h1 = model(image) 
        embedding = h1
        embeddings_matrix = torch.cat((embeddings_matrix, embedding))

  embeddings_test = embeddings_matrix.cpu().numpy()

  if os.path.exists('data.h5'):
    os.remove('data.h5')

  f = h5py.File('data.h5', 'w')
  f.create_dataset("embeddings", data=embeddings_test)
  dataset_scann = f['embeddings']
  normalized_dataset = dataset_scann / np.linalg.norm(dataset_scann, axis=1)[:, np.newaxis]
  searcher = scann.scann_ops_pybind.builder(normalized_dataset, 50, "dot_product").tree(num_leaves = int(np.sqrt(len(dataset_scann))), num_leaves_to_search = 10).score_brute_force().build() 
  neighbors, distances = searcher.search_batched(normalized_dataset)

  #gets label for each image by index
  def labelLookup(index):
    return dataset.labels[index]

  lookup = np.vectorize(labelLookup)
  result_array = lookup(neighbors)

  neighbor_rank = 1

  array = confusion_matrix(result_array[:,0], result_array[:,neighbor_rank], normalize='true')
  for i, r in enumerate(array):
    acc_row = r[i]/ sum(r)

  v = len(array)
  df_cm = pd.DataFrame(array, range(v), range(v))
  plt.figure(figsize=(10,7))
  plt.title('Rank 1 on embeddings using validation transform')
  sn.set(font_scale=0.5) # for label size
  res = sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}) # font size
  figure = res.get_figure()    
  figure.savefig(f'{save_path}/rank1_NN_heatmap.png', dpi=400)
  plt.clf()
  plt.cla()
  reference_image_classes = result_array[:, 0]
  accs_by_rank = []
  ncols = result_array.shape[1]
  nrows = result_array.shape[0]


  for i in range(1, ncols):
    accs_by_rank.append(np.sum(reference_image_classes == result_array[:, i])/nrows)

  plt.rc('ytick', labelsize=10)
  plt.plot(range(1, ncols), accs_by_rank)
  plt.xlabel('Nearest Neighbor Rank')
  plt.ylabel('Percent in Reference Image Class')
  plt.title('SimCLR (All Data) Similarity Searching')
  plt.savefig(f'{save_path}/NN_acc_by_rank.png', dpi=400)

  if os.path.exists('data.h5'):
      os.remove('data.h5')

def cli_main():
    
    parser = ArgumentParser()
    parser.add_argument("--MODEL_PATH", type=str, help="path to .pt file containing SSL-trained SimCLR Resnet18 Model")
    parser.add_argument("--DATA_PATH", type = str, help = "path to data. If folder already contains validation data only, set val_split to 0")
    parser.add_argument("--val_split", default = 0.2, type = float, help = "amount of data to use for validation as a decimal")
    parser.add_argument("--image_type", default="tif", type=str, help="extension of image for PIL to open and parse - i.e. jpeg, gif, tif, etc. Only put the extension name, not the dot (.)")
    parser.add_argument("--image_embedding_size", default=128, type=int, help="size of image representation of SIMCLR")
    parser.add_argument("--image_size", default = 128, type=int, help="height of square image to pass through model")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")

    args = parser.parse_args()
    MODEL_PATH = args.MODEL_PATH
    DATA_PATH = args.DATA_PATH
    image_size = args.image_size
    image_type = args.image_type
    embedding_size = args.image_embedding_size
    val_split = args.val_split
    gpus = args.gpus

    #testing
    # MODEL_PATH = '/content/models/SSL/SIMCLR_SSL_0.pt'
    # DATA_PATH = '/content/UCMerced_LandUse/Images'
    # image_size = 128
    # image_type = 'tif'
    # embedding_size = 128
    # val_split = 0.2
    # gpus = 1

    

        # #gets dataset. We can't combine since validation data has different transform needed
    train_dataset = FolderDataset(DATA_PATH, validation = False, 
                                  val_split = val_split, 
                                  transform = SimCLRTrainDataTransform(image_size), 
                                  image_type = image_type
                                  ) 
    


    print('Training Data Loaded...')
    val_dataset = FolderDataset(DATA_PATH, validation = True,
                                val_split = val_split,
                                transform = SimCLREvalDataTransform(image_size),
                                image_type = image_type
                                )
    

    print('Validation Data Loaded...')

    #load model
    num_samples = len(train_dataset)

    #init model with batch size, num_samples (len of data), epochs to train, and autofinds learning rate
    model = SimCLR(arch = 'resnet18', batch_size = 1, num_samples = num_samples, gpus = gpus, dataset = 'None') #
    
    model.encoder = resnet18(pretrained=False, first_conv=model.first_conv, maxpool1=model.maxpool1, return_all_feature_maps=False)
    model.projection = Projection(input_dim = 512, hidden_dim = 256, output_dim = embedding_size) #overrides

    model.load_state_dict(torch.load(MODEL_PATH))
    

    model.cuda()
    print('Successfully loaded your model for evaluation.')

    #running eval on training data
    save_path = f"{MODEL_PATH[:-3]}/Evaluation/trainingMetrics"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    eval_embeddings(model, train_dataset, save_path)
    print('Training Data Evaluation Complete.')
    #running eval on validation data
    save_path = f"{MODEL_PATH[:-3]}/Evaluation/validationMetrics"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    eval_embeddings(model, val_dataset, save_path)
    print('Validation Data Evaluation Complete.')
    print(f'Please check {MODEL_PATH[:-3]}/Evaluation/ for your results')

if __name__ == '__main__':
    cli_main()
