import umap.distances as dists
from umap.umap_ import UMAP, simplicial_set_embedding
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
from sklearn.utils import check_array

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
        approx_method = "mean_dist",
        output_type = "default",
        **params
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_components = n_components
        self.min_dist = min_dist
        self.__dict__.update(params)
        self.umap_params = self.__dict__.copy()
        
        self.random_state = random_state
        self.n_projections = n_projections
        self.approx_method = approx_method
        self.output_type = output_type

    def get_dist_matrix(self, X):
        return pairwise_distances(X, metric=dists.named_distances[self.metric])
    
    def get_mode_neighbors(self, X):
        N = X.shape[0]
        modas = []
        for i in range(N):
            x = X[i,:].copy()
            ids = np.arange(N)
            moda = sorted(ids, key=lambda i: x[i])
            modas.apped(moda[:self.n_neighbors])
        return modas        

    def fit_transform(self, X): # method=["mean_dist", "comment_neighbor_mode"]
        # if self.approx_method == "mean_dist":
        self.dists = np.zeros((self.n_projections, X.shape[0], X.shape[0]))
        rng = np.random.default_rng(self.random_state)
        seeds = rng.choice(99_999_999, size=self.n_projections, replace=False)
        for i in range(self.n_projections):
            self.umap_params['random_state'] = seeds[i] 
            map_reduce = UMAP(**self.umap_params)
            new_X = map_reduce.fit_transform(X)
            self.dists[i,:,:] = self.get_dist_matrix(new_X)
        if self.output_type == "precomputed":
            return self.dists.mean(axis=0)
        elif self.output_type == "default":
            media_matrix = self.dists.mean(axis=0) # (n, n)
            dist = np.abs(media_matrix-self.dists).mean(axis=0) # (n_proj,)
            self.umap_params['random_state'] = seeds[np.argmin(dist)] 
            return UMAP(**self.umap_params).fit_transform(X) # (n, n_components) 
        elif self.output_type == "simplicial":
            self.embedding_, aux_data = simplicial_set_embedding(
                X,
                self.dists.mean(axis=0),
                self.a,
                self.b,
                self.repulsion_strength,
                self.negative_sample_rate,
                0 if self.n_epochs is None else self.n_epochs,
                self.init,
                self.random_state,
                self._input_distance_func,
                self._metric_kwds,
                self.densmap,
                self._densmap_kwds,
                self.output_dens,
                self._output_distance_func,
                self._output_metric_kwds,
                self.output_metric in ("euclidean", "l2"),
                self.random_state is None,
                self.verbose,
                tqdm_kwds=self.tqdm_kwds,
            )
            return self.embedding_


        # elif self.approx_method == "comment_neighbor_mode":
        #     nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="precomputed")
        #     rng = np.random.default_rng(self.random_state)
        #     seeds = rng.choice(99_999_999, size=self.n_projections, replace=False)
        #     modas = []
        #     for i in range(self.n_projections):
        #         self.umap_params['random_state'] = seeds[i] 
        #         map_reduce = UMAP(**self.umap_params)
        #         new_X = map_reduce.fit_transform(X)
        #         if modas == []:
        #             modas = self.get_mode_neighbors(new_X)
        #         else:
        #             atual_moda = self.get_mode_neighbors(new_X)
        #             for i in range(X.shape[0]):
        #                 modas[i] += atual_moda[i]
        #     escolhidos = [mode(e).mode[:self.n_neighbors] for e in modas]
        #     counts = [mode(e).counts[:self.n_neighbors] for e in modas]
            # compute probability
            
                    # rodar em todas posições distribuindo os vizinhos
                    # mais prox
                    # fazer adaptação da probabilidade:
                        # priori uniforme
                    # rodar testes
