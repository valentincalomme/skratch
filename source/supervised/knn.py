"""Implementation of KNN"""

from collections import Counter

import numpy as np

from utils.distances import euclidean, cdist


class KNN:

    def __init__(self, k=1, weighted=False):

        self.k = k
        self.weighted = weighted

    def fit(self, X, y):

        self.X_ = X
        self.y_ = y

        return self

    def update(self, X, y):

        self.X_ = np.concatenate((self.X_, X))
        self.y_ = np.concatenate((self.y_, y))

        return self

    def predict(self, X):

        predictions = []

        for x in X:

            neighbours, distances = self._get_neighbours(x)

            prediction = self._vote(neighbours, distances)

            predictions.append(prediction)

        return np.array(predictions)

    def _get_neighbours(self, x):

        distances = np.array([self._distance(x, x_) for x_ in self.X_])
        indices = np.argsort(distances)[:self.k]

        return self.y_[indices], distances[indices]

    def _distance(self, a, b):

        return euclidean(a, b)

    def _get_weights(self, distances):

        weights = np.ones_like(distances, dtype=float)

        if self.weighted:
            if any(distances == 0):
                weights[distances != 0] = 0
            else:
                weights /= distances

        return weights

    def _vote(self, targets, distances):
        raise NotImplementedError("KNN requires a _vote function")


class KNN_Classifier(KNN):

    def _vote(self, classes, distances):

        weights = self._get_weights(distances)

        weighted_frequencies = {c: np.sum(weights[classes == c]) for c in list(set(classes))}

        return max(classes, key=weighted_frequencies.get)


class KNN_Regressor(KNN):

    def _vote(self, targets, distances):

        weights = self._get_weights(distances)

        return np.sum(weights * targets) / np.sum(weights)
