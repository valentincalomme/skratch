import numpy as np
import copy

from unsupervised.kmeans import KMeans
from utils.distances import manhattan


class KMedians(KMeans):

    def __init__(self, k=3, seed=0, n_runs=10):

        super().__init__(k=k, seed=seed, n_runs=n_runs)

    def _distance(self, a, b):

        return manhattan(a, b)

    def _compute_centroids(self, X, labels):

        centroids = []

        for i in range(self.k):

            centroid = np.array([np.median(dim) for dim in X[labels == i].T])
            centroids.append(centroid)

        return np.array(centroids)