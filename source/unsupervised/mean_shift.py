"""Mean shift"""

import numpy as np

from utils.kernels import RBF
from utils.distances import euclidean


class MeanShift:

    def __init__(self, bandwith=1, tol=1E-7):

        self.bandwith = bandwith
        self.tol = 1 - tol
        self.kernel = RBF(gamma=self.bandwith)

    def _compute_labels(self, X, centers):

        labels = []

        for x in X:

            distances = np.array([euclidean(x, center) for center in centers])
            label = np.argmin(distances)
            labels.append(label)

        _, labels = np.unique(labels, return_inverse=True)
        return np.array(labels, dtype=np.int)

    def predict(self, X):

        labels = self._compute_labels(X, self.cluster_centers_)

        return labels

    def fit(self, X):

        for labels, centers in self._fit(X):

            self.labels_ = labels
            self.cluster_centers_ = centers

        return self

    def _fit(self, X):

        old_centers = np.array([])
        new_centers = X
        labels = -np.ones(len(X))  # -1 represents an "orphan"

        while not self._has_converged(old_centers, new_centers):

            yield labels, new_centers

            old_centers = new_centers
            new_centers = []

            for center in old_centers:

                shifted_center = self._shift(center, X)
                new_centers.append(shifted_center)

            new_centers = self._merge_centers(new_centers)
            labels = self._compute_labels(X, new_centers)

    def _shift(self, x, X):

        densities = [self.kernel(x, x_) for x_ in X]

        shifted_center = np.average(X, weights=densities, axis=0)

        return shifted_center

    def _merge_centers(self, centers):

        centers = np.unique(centers, axis=0)
        new_centers = []

        for c in centers:
            distances = np.array([self.kernel(c, c_) for c_ in centers])
            new_centers.append(np.mean(centers[distances > self.tol], axis=0))

        centers = np.unique(new_centers, axis=0)

        return centers

    def _has_converged(self, old, new):

        if len(old) == len(new):

            for i in range(len(new)):
                if self.kernel(old[i], new[i]) < 1.0:
                    return False

            return True
        else:
            return False
