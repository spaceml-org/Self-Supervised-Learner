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


def plot_umap(feature_list, filenames , path, n_neighbors=20, count = 0):
  # feature_list = feature_list.detach().numpy()
  class_id = []
  for _ in filenames:
    class_id.append(_.split("/")[6])
  num_points = dict((x,class_id.count(x)) for x in set(class_id)))
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
  for _ in tqdm(range(n - 1)) :
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

def get_embeddings(ckpt_path,dataloader):

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
    os.makedirs(os.path.join(dir, 'train'))

  for x in paths:
    if x.split("/")[6] not in os.listdir(os.path.join(dir,"train")):
      os.mkdir(os.path.join(os.path.join(dir,"train"),x.split("/")[6]))
      shutil.copy(x, os.path.join(os.path.join(dir,"train"),x.split("/")[6]))
    else:
      shutil.copy(x, os.path.join(os.path.join(dir,"train"),x.split("/")[6]))
  print('Subset moved to directory. Dataset abiding formats created successfully')

def farthest_point(embeddings):
  import scipy.spatial.distance as dist
  centroid = sum(embeddings)/len(embeddings)
  distances = [dist.euclidean(i, centroid) for i in embeddings]

  return distances.index(max(distances))

def class_distrib(filenames, metric = 'count'):
    filenames = paths.list_images(os.path.join(filenames, 'train'))
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

def animate_umap(path, fps= 1, format = 'gif'):
  images = []
  files = natsorted(list(paths.list_images(path)))
  for filename in files:
      images.append(imageio.imread(filename))
  output = path + '_umap_seq.' + format
  if os.path.exists(output):
    os.remove(output)
  imageio.mimsave(output, images, fps=1)
