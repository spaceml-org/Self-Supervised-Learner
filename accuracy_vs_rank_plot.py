import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from tqdm.notebook import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def transform(im: PIL.Image.Image):
    return torch.tensor(np.array(im)).permute(2,0,1).float().cuda()

# The only inputs necessary for this function are the model, the model embedding size, the directory to the images, and the upper limit on the rank for the plot
# Make sure to load the model in with the data before you call this function. Also make sure you have the right size for the model embeddings!

def rank_vs_accuracy_plot(path,model,embedding_size,upper_bound):
  dataset = ImageFolder(path, transform = transform)
  dataloaded = DataLoader(dataset, shuffle = False, batch_size = 1)
  
  embeddings =  torch.empty(size = (0,embedding_size))

  with torch.no_grad():
    for image, y in tqdm(dataloaded):
      output = model(image).cpu()
      embeddings = torch.vstack((embeddings, output))
      
  #getting embedding distances    
  embedding_distance = cosine_similarity(embeddings)
  np.fill_diagonal(embedding_distance,0)
  
  furthest = embedding_distance.argsort()
  nearest = np.flip(furthest, axis = 1)
  
  #create a dictionary mapping the index to a class
  im_list = dataset.imgs
  class_lookup = {}
  for i in range(len(im_list)):
    class_lookup[i] = im_list[i][1]

  #map onto nearest images by index
  class_sim = np.vectorize(class_lookup.get)(nearest)
  
  reference_class_labels = class_sim[:, -1]
  k_correct =[]

  for n in tqdm(range(1, upper_bound+1)):
    top_n_closest = class_sim[:, :n]

    #sum where correctly in same class
    correct = 0
    index = 0
    for row in top_n_closest:
      correct += sum(np.equal(row, reference_class_labels[index]))
      index += 1

    avg_correct = correct/(index*n)
    n_correct.append(avg_correct)
  
  plt.figure(figsize=(15,15))
  plt.plot(np.arange(1,upper_bound+1),n_correct,'-')
  plt.title('Top-N in Same Class')
  plt.xlabel('N',fontsize=24)
  plt.ylabel('% of Same Class',fontsize=24)
  plt.ylim(0,100)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.show()
