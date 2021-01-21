import umap
from matplotlib import colors 
import matplotlib.pyplot as plt
import random 
import os
import shutil
import numpy as np
from tqdm import tqdm
from imutils import paths
from natsort import natsorted
from ssl_dali_distrib import SIMCLR
from sklearn.preprocessing import LabelEncoder
import matplotlib.animation as animation
import imageio
import torch
from tqdm.notebook import tqdm
import PIL.Image as Image
from torchvision import transforms

def plot_umap(feature_list, filenames , path, n_neighbors=20, count = 0):
  # feature_list = feature_list.detach().numpy()
  class_id = []
  for _ in filenames:
    class_id.append(_.split("/")[-2])
  num_points = dict((x,class_id.count(x)) for x in set(class_id))
  txt = ''
  for i in num_points.keys():
    txt += i + ':' + str(num_points[i]) + " "
  le = LabelEncoder()
  class_labels = le.fit_transform(class_id)
  # print("Classes: ",le.classes_)
  fit = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric='euclidean')

  u = fit.fit_transform(feature_list)
  color_map = plt.cm.get_cmap('tab20b_r')
  scatter_plot= plt.scatter(u[:,0], u[:, 1], c=class_labels, cmap = color_map)
  plt.title('UMAP embedding of random colours. Iteration ' + str(count+1));
  plt.colorbar(scatter_plot)
#   plt.text(.5, .05, txt, ha='center')

  # if not os.path.isdir("./graphs/"):
  #   print("Creating directory to store graphs")
  #   os.mkdir("./graphs/")
  
  fname = path + 'UMAP_DA_Itr_' + str(count+1) + '.png'
  plt.savefig(fname)
  plt.show();

def min_max_diverse_embeddings(n , filenames, feature_list, i = None) :
  if len(feature_list) != len(filenames) or len(feature_list) == 0 :
      return 'Data Inconsistent'
  n = int(n * len(feature_list))
  print("Len of Filenames and Feature List for sanity check:",len(filenames),len(feature_list))
  # print(filenames[0],feature_list[0])
  # filenames = filenames.detach().numpy()

  # feature_list = feature_list.detach().numpy()
  filename_copy = filenames.copy()
  set_input = feature_list.copy()
  set_output = []
  filename_output = []
  #random.seed(SEED)
  idx = 0
  if i is None: 
      idx = random.randint(0, len(set_input) -1)
  else:
      idx = i
  set_output.append(set_input[idx])
  filename_output.append(filename_copy[idx])
  min_distances = [1000] * len(set_input)
  # maximizes the minimum distance
  for _ in tqdm(range(n - 1)):
      for i in range(len(set_input)) :
          # distances[i] = minimum of the distances between set_output and one of set_input
          dist = np.linalg.norm(set_input[i] - set_output[-1])
          if min_distances[i] > dist :
              min_distances[i] = dist
      inds = min_distances.index(max(min_distances))
      set_output.append(set_input[inds])
      filename_output.append(filename_copy[inds])
      del set_input[inds]
      del filename_copy[inds]
      del min_distances[inds]
  return filename_output, set_output, min_distances

def get_embeddings_test(ckpt, PATH, size= 256):
  '''
  ckpt : checkpoint path 
  PATH : Dataset path
  '''
  model = SIMCLR.load_from_checkpoint(ckpt)
  model.eval()
  model.cuda()
  t= transforms.Resize((size, size))
  embedding_matrix = torch.empty(size= (0, model.embedding_size)).cuda()
  model = model.encoder
  ims = []
  for folder in os.listdir(PATH):
    for im in os.listdir(f'{PATH}/{folder}'):
      ims.append(f'{PATH}/{folder}/{im}')
  for f in tqdm(ims):
    with torch.no_grad():
      im = Image.open(f).convert('RGB')
      im = t(im)
      im = np.asarray(im).transpose(2, 0, 1)
      im = torch.Tensor(im).unsqueeze(0).cuda()
      embedding = model(im)[0]
      embedding_matrix = torch.vstack((embedding_matrix, embedding))
  print('Embedding Shape', embedding_matrix.shape)
  return embedding_matrix.detach().cpu().numpy()

def get_embeddings(ckpt_path, PATH):

  embedding = []
  labels = []
  simclr_model = SIMCLR.load_from_checkpoint(ckpt_path)
  for step,(x,y) in enumerate(dataloader):
    embedding.extend(simclr_model(x).detach().numpy())
    # labels.extend(y.detach().numpy())
  return embedding

def n_random_subset(n, paths):  
  
  n = int(n * len(paths))
  datapoints = random.sample(paths, n)
  # random_points = [paths[i] for i in datapoints]
  
  return datapoints

##TODO add class distribution to the umap plot tomorrow. 


def prepare_dataset(dir, paths): 

  try: 
    os.makedirs(os.path.join(dir, 'train'))
  except:
    shutil.rmtree(dir)
    os.mkdir(dir)
    # os.makedirs(os.path.join(dir, 'train'))

  for x in paths:
    # if x.split("/")[-2] not in os.listdir(os.path.join(dir,"train")):
    if x.split("/")[-2] not in os.listdir(dir):
      # os.mkdir(os.path.join(os.path.join(dir,"train"),x.split("/")[-2]))
      os.mkdir(os.path.join(dir,x.split("/")[-2]))
      shutil.copy(x, os.path.join(os.path.join(dir,x.split("/")[-2])))
    else:
      shutil.copy(x, os.path.join(os.path.join(dir,x.split("/")[-2])))
  print('Subset moved to directory. Dataset abiding formats created successfully')

def farthest_point(embeddings):
  import scipy.spatial.distance as dist
  centroid = sum(embeddings)/len(embeddings)
  distances = [dist.euclidean(i, centroid) for i in embeddings]

  return distances.index(max(distances))

def class_distrib(filenames, metric = 'count'):
    filenames = paths.list_images(filenames)
    classes = {}
    for x in filenames:
      if x.split('/')[-2] in classes:
        classes[x.split('/')[-2]] += 1
      else:
        classes[x.split('/')[-2]] = 1
    if metric == 'count':
      classes_dict = {}
      for _ in classes.keys():
        print(_ , '= ', classes[_])
        classes_dict[_] = classes[_]
      x = classes_dict
    elif metric == 'stddev':
      x = np.std(list(classes.values()))
      print('Standard Deviation between classes ', x)

    elif metric == 'difference':
      x = max(list(classes.values()))- min(list(classes.values()))
      print('Difference between max and minimum values ', x)
    return x

def plot_metrics(metric, metric_array):
  plt.title(metric+ " across time" );
  plt.plot(metric_array)
  fname = "./graphs/" + metric + '_graph'+ '.png'
  plt.savefig(fname)
  plt.show();


def animate(path, fps= 1, format = 'gif', method = 'tsne'):
  images = []
  files = natsorted(list(paths.list_images(path)))
  for filename in files:
      images.append(imageio.imread(filename))
  output = path + '_umap_seq.' + format
  if os.path.exists(output):
    os.remove(output)
  imageio.mimsave(output, images, fps=1)

