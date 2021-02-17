import numpy as np
import pandas as pd
import faiss
import time
from pathlib import Path
import sys
sys.path.insert(0, './SpaceForceDataSearch')
from torchvision import transforms
from ssl_dali_distrib import SIMCLR 
from finetuner_dali_distrib import finetuner
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score
from torchvision.datasets import ImageFolder

#####ALL HELPER FUNCTIONS BELOW
def faiss_nearest_neighbors(query, num_candidates = None, dataset = None, index = None):
    '''
    Finds nearest neighbors
    dataset: numpy matrix of shape N x (embedding size)
    query: numpy matrix of Q x (embedding size)
    num_candidates (int): number of candidates to find
    searcher (faiss.swigfaiss.IndexFlatL2): a faiss prebuilt searcher. If argument passed, dataset and num_candidates argument is not used
    returns:
    neighbors (numpy array): numpy matrix of indices of nearest neighbors of shape Q x num_candidates
    '''
    print('Retrieving Similar Embeddings')
    if index is None:
        start = time.time()
        index = faiss.IndexFlatL2(dataset.shape[1])
        index.add(dataset.astype('float32'))
        print('Time Taken to build faiss database: ', time.time()-start)
    start = time.time()
    D, I = index.search(query.astype('float32'), num_candidates)
    print('Time Taken to find neighbors: ', time.time()-start)
    #unique candidates
    candidates = pd.unique(I.flatten()[D.flatten().argsort()])
def load_checkpoint(MODEL_PATH):
    #expects a checkpoint path, not an encoder
    try:
        model = finetuner.load_from_checkpoint(MODEL_PATH)
        is_classifier = True
    except:
        model = SIMCLR.load_from_checkpoint(MODEL_PATH)
        is_classifier = False
    return model, is_classifier
def get_matrix(MODEL_PATH, DATA_PATH):
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()
    t = transforms.Compose([
                            transforms.Resize((64,64)),
                            transforms.Lambda(to_tensor)
                            ])
    dataset = ImageFolder(DATA_PATH, transform = t)
    model, is_classifier = load_checkpoint(MODEL_PATH)
    model.eval()
    model.cuda()
    if is_classifier:
      size = model.num_classes
    else:
      size = model.embedding_size
    with torch.no_grad():
        data_matrix = torch.empty(size = (0, size)).cuda()
        bs = 32
        if len(dataset) < bs:
          bs = 1
        loader = DataLoader(dataset, batch_size = bs, shuffle = False)
        for batch in loader:
            x = batch[0].cuda()
            embeddings = model(x)
            data_matrix = torch.vstack((data_matrix, embeddings))
    return data_matrix.cpu().detach().numpy(), dataset.imgs
    return candidates[:num_candidates]
