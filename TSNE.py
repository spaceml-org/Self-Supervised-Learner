import pickle
from tqdm import notebook
import random
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import PIL
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.preprocessing import LabelEncoder

class TSNE_visualiser: 
    def __init__(self, feature_list, filenames):
        '''
        params:
        feature_list : Embeeddings list
        filenames: filenames for the images in the embeddings list
        '''
        self.feature_list = feature_list
        self.filenames = filenames
        self.labels = []
        for _ in filenames:
            self.labels.append(_.split("/")[6])

    # return train_data_reshaped, labels, feature_list
 
    def knn_cluster(self, feature_list):
    '''
    params: 
    feature_list : list contraining all output embeddings
    '''
        neighbors=  NearestNeighbors(n_neighbors= 5, algorithm= 'brute', metric= 'euclidean').fit(feature_list)  
        distances, indices = neighbors.kneighbors(feature_list)
        print("Median distance between all photos: ", np.median(distances))
        print("Max distance between all photos: ", np.max(distances))
        print("Median distance among most similar photos: ", np.median(distances[:, 2]))
        return neighbors , distances, indices

    def fit_tsne(self, feature_list, perplexity= 30, n_jobs= 4):
    '''
    Fits TSNE for the input embeddings
    feature_list: ssl embeddings
    perplexity : hyperparameter that determines how many images are close to each other in a cluster
    n_jobs : number of jobs to be run concurrently. 
    '''
        n_components = 2
        verbose = 1
        perplexity = perplexity
        n_iter = 1000
        metric = 'euclidean'
        n_jobs= n_jobs

        time_start = time.time()
        tsne_results = TSNE(n_components=n_components,
                            verbose=verbose,
                            perplexity=perplexity,
                            n_iter=n_iter,
                            n_jobs= n_jobs,
                            metric=metric).fit_transform(feature_list)

        print('T-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
        return tsne_results
    
    def scatter_plot(self, tsne_results, labels, path, count = 0):
    '''
    Plots a scatter plot for the given TSNE fit variable
    '''
        le = LabelEncoder()
        class_labels = le.fit_transform(labels)
        color_map = plt.cm.get_cmap('tab20b_r')
        scatter_plot = plt.scatter(tsne_results[:, 0],
                                tsne_results[:, 1],
                                c=labels,
                                cmap=color_map)
        
        plt.colorbar(scatter_plot)
        fname = path + 'TSNE_Scatter_DA_Itr_' + str(count+1) + '.png'
        plt.savefig(fname)
        plt.show()

    def plot_images_in_2d(self, x, y, image_vectors, axis=None, zoom=1):
    '''
    Helper function, do not call. 
    params:
    x, y : TSNE variables
    image_vectors: images in the dataset
    '''
        if axis is None:
            axis = plt.gca()
        x, y = np.atleast_1d(x, y)
        for x0, y0, image_path in zip(x, y, image_vectors):
            image_path= image_path.numpy()
            z = (image_path * 255).astype(np.uint8)
            image = Image.fromarray(z)
            image.thumbnail((200, 200), Image.ANTIALIAS)
            img = OffsetImage(image, zoom=zoom)
            anno_box = AnnotationBbox(img, (x0, y0),
                                    xycoords='data',
                                    frameon=False)
            axis.add_artist(anno_box)
        axis.update_datalim(np.column_stack([x, y]))
        axis.autoscale()

  def show_tsne(self, x, y, images, path, count = 0):
        fig, axis = plt.subplots()
        fig.set_size_inches(22, 22, forward=True)
        self.plot_images_in_2d(x, y, images, zoom=0.3, axis=axis)
        fname = path + 'TSNE_Image_Plot_DA_Itr_' + str(count+1) + '.png'
        plt.savefig(fname)
        plt.show() 

  def tsne_to_grid_plotter_manual(self, x, y, selected_filenames, path, count):
    '''
    TSNE visualiser with evenly spaced out images
    params:
    x, y : TSNE variables
    selected_filenames: images in the dataset
    '''
        S = 2000
        s = 100
        x = (x - min(x)) / (max(x) - min(x))
        y = (y - min(y)) / (max(y) - min(y))
        x_values = []
        y_values = []
        filename_plot = []
        x_y_dict = {}
        for i, image_path in enumerate(selected_filenames):
            a = np.ceil(x[i] * (S - s))
            b = np.ceil(y[i] * (S - s))
            a = int(a - np.mod(a, s))
            b = int(b - np.mod(b, s))
            if str(a) + "|" + str(b) in x_y_dict:
                continue
            x_y_dict[str(a) + "|" + str(b)] = 1
            x_values.append(a)
            y_values.append(b)
            filename_plot.append(image_path)
        fig, axis = plt.subplots()
        fig.set_size_inches(22, 22, forward=True)
        self.plot_images_in_2d(x_values, y_values, filename_plot, zoom=.58, axis=axis)
        fname = path + 'TSNE_GridPlot_DA_Itr_' + str(count+1) + '.png'
        plt.savefig(fname)
        plt.show()