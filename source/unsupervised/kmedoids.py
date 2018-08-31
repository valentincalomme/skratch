import numpy as np
import copy

from unsupervised.kmeans import KMeans
from utils.distances import pdist, euclidean, manhattan, cosine


class KMedoids(KMeans):

    def _distance(self, a, b):

        return cosine(a, b)

    def _compute_centroids(self, X, labels):

        if not hasattr(self, "distances"):
            self.distances = pdist(X, self._distance)

        centroids = []

        for i in range(self.k):

            distances = self.distances[np.ix_(labels == i, labels == i)]
            within_cluster_sum_of_distances = np.sum(distances, axis=0)

            centroid = X[labels == i][np.argmin(within_cluster_sum_of_distances)]

            centroids.append(centroid)

        return np.array(centroids)
