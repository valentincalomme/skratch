import numpy as np

from utils.distances import pdist, euclidean


class DBSCAN:

    def __init__(self, min_samples, epsilon):

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

        index = 0

        for i, x in enumerate(X):

            if self.labels_[i] == 0:  # point is undefined

                neighbours = self._get_neighbours(i)

                if len(neighbours) >= self.min_samples:

                    index += 1
                    self.labels_[i] = index

                    for neighbour in neighbours:

                        if self.labels_[neighbour] <= 0:  # undefined or noise
                            self.labels_[neighbour] = index

                            new_neighbours = np.array([n for n in self._get_neighbours(neighbour)
                                                       if n not in neighbours])

                            neighbours = np.hstack((neighbours, new_neighbours))

                else:
                    self.labels_[i] = -1  # defined as noise

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

if __name__ == "__main__":

    X = np.random.rand(10, 2)

    clust = DBSCAN(min_samples=3, epsilon=0.5)
    clust.fit(X)