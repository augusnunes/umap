import umap.distances as dists
import umap.umap_ import UMAP 
import numpy as np
from sklearn.metrics import pairwise_distances

class ApproximateUMAP:
    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        min_dist=0.1,
        random_state=None,
        # Approximate params
        n_projections = 20,
        **params
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_components = n_components
        self.min_dist = min_dist
        self.random_state = random_state
        self.__dict__.update(params)
        self.umap_params = self.__dict__.copy()
        
        self.n_projections = n_projections


    def get_dist_matriz(self):
        return pairwise_distances(self._tmp_X, metric=dists.named_distances[self.metric])
    
    def fit_transform(self, X): # method=["mean_dist", "comment_neighbor_mode"]
        self.dists = np.zeros((self.n_projections, X.shape[0], X.shape[0]))
        for i in range(self.n_projections):
            self.umap_params['random_state'] = np.random.random() 
            map_reduce = UMAP(**self.umap_params)
            new_X = map_reduce.fit_transform(X)
            self.dists[i,:,:] = get_dist_matriz(new_X)
        return self.dists.mean(axis=0)


