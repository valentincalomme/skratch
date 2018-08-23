import numpy as np

from utils.distances import pdist, euclidean

UNDEFINED = 0
NOISE = -1


class DBSCAN:

    def __init__(self, eps=0.5, min_samples=5):

        self.min_samples = min_samples
        self.epsilon = epsilon

    def _distance(self, a, b):

        return euclidean(a, b)

    def fit(self, X, y=None):

        self.labels_ = self._initialize_labels(X, index=0)
        self.distances_ = self._compute_distances(X)

        for labels in self._dbscan(X):
            print(labels)

        return self

    def _dbscan(self, X):

        cluster_index = UNDEFINED

        for i, x in enumerate(X):

            if self.labels_[i] == UNDEFINED:

                neighbours = self._get_neighbours(i)

                if len(neighbours) < self.min_samples:

                    self.labels_[i] = NOISE  # defined as noise

                else:

                    cluster_index += 1  # create a new cluster

                    self.labels_[i] = cluster_index  # assign the core point to the cluster

                    for neighbour in neighbours:

                        if self.labels_[neighbour] == UNDEFINED or self.labels_[neighbour] == NOISE:

                            self.labels_[neighbour] = cluster_index

                            new_neighbours = np.array([n for n in self._get_neighbours(neighbour)
                                                       if n not in neighbours])

                            neighbours = np.hstack((neighbours, new_neighbours))

            yield self.labels_

    def _get_neighbours(self, index):

        indices, = np.where(self.distances_[index] < self.epsilon)

        return indices[indices != index]

    def _initialize_labels(self, X, index):

        return np.full(len(X), index)

    def _compute_distances(self, X):

        return pdist(X, self._distance)

    def predict(self, X):

        return
