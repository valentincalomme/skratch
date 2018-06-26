import numpy as np
import copy

from unsupervised.kmeans import KMeans
from utils.distances import pdist, euclidean, manhattan, cosine


class KMedoids(KMeans):

    def __init__(self, k=3, seed=0, n_runs=10):

        super().__init__(k=k, seed=seed, n_runs=n_runs)

    def _distance(self, a, b):

        return manhattan(a, b)

    def _compute_centroids(self, X, labels):

        centroids = []

        for i in range(self.k):

            distances = pdist(X[labels == i], self._distance)
            within_cluster_sum_of_distances = np.sum(distances, axis=0)

            centroid = X[labels == i][np.argmin(within_cluster_sum_of_distances)]

            centroids.append(centroid)

        return np.array(centroids)