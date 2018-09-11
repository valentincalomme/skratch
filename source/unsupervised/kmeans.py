import copy

import numpy as np

from utils.distances import euclidean


class KMeans:

    def __init__(self, k=3, seed=None, n_runs=10, max_iters=300):

        self.max_iters = max_iters
        self.k = k
        self.rnd = np.random.RandomState(seed)
        self.n_runs = n_runs

    def _distance(self, a, b):

        return euclidean(a, b)

    def _inertia(self, X, centroids, labels):

        distances = []

        for i, centroid in enumerate(centroids):

            distances.extend([self._distance(x, centroid)**2 for x in X[labels == i]])

        return np.sum(distances)

    def _compute_centroids(self, X, labels):

        centroids = []

        for i in range(self.k):

            centroid = np.mean(X[labels == i], axis=0)
            centroids.append(centroid)

        return np.array(centroids)

    def _compute_labels(self, X, centroids):

        labels = []

        for x in X:

            distances = [self._distance(x, centroid) for centroid in centroids]
            label = np.argmin(distances)
            labels.append(label)

        return np.array(labels)

    def predict(self, X):

        return self._compute_labels(X, self.centroids_)

    def fit(self, X, y=None):

        self.inertia_ = float('inf')  # initialize as worst possible inertia

        for run in range(self.n_runs):

            for i, (c, l) in enumerate(self._fit(X, y)):
                centroids, labels = c, l

                if i > self.max_iters:
                    break

            inertia = self._inertia(X, centroids, labels)

            if inertia < self.inertia_:

                self.inertia_ = inertia
                self.centroids_ = centroids
                self.labels_ = labels

        return self

    def _fit(self, X, y=None):

        centroids = self._initialize_centroids(X)
        labels = self._compute_labels(X, centroids)
        old_labels = np.full_like(labels, -1)

        while any(old_labels != labels):

            old_labels = labels
            centroids = self._compute_centroids(X, labels)
            labels = self._compute_labels(X, centroids)

            yield centroids, labels

    def _initialize_centroids(self, X):

        X_ = X.copy()
        self.rnd.shuffle(X_)
        return X_[:self.k]

    def _kmeans_pp(self, X):

        centroids = []

        weights = np.ones(len(X))
        weights /= weights.sum()

        for k in range(self.k):

            centroid = X[np.random.choice(np.arange(len(X)), 1, p=weights)[0], :]

            centroids.append(centroid)

            distances = np.array([self._distance(centroid, x) for x in X])

            weights = distances**2
            weights /= weights.sum()

        return centroids