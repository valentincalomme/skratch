"""Implementation of KNN"""

from collections import Counter

import numpy as np

from utils.distances import euclidean


class KNN:

    def __init__(self, k, weighted=False):

        self.k = k
        self.weighted = weighted  # Whether or not to use weighted distances

    def fit(self, X, y):

        self.X_ = X
        self.y_ = y

        return self

    def update(self, X, y):

        self.X_ = np.concatenate((self.X_, X))  # append new data to the already existing data
        self.y_ = np.concatenate((self.y_, y))  # append new data to the already existing data

        return self

    def predict(self, X):

        predictions = []

        for x in X:

            neighbours, distances = self._get_neighbours(x)

            prediction = self._vote(neighbours, distances)  # Make a prediction

            predictions.append(prediction)

        return np.array(predictions)

    def _get_neighbours(self, x):

        distances = np.array([self._distance(x, x_) for x_ in self.X_])
        indices = np.argsort(distances)[:self.k]  # keep the indices of the k-nearest neighbours

        return self.y_[indices], distances[indices]

    def _distance(self, a, b):

        return euclidean(a, b)

    def _get_weights(self, distances):

        weights = np.ones_like(distances, dtype=float)  # By default, all neighbours are uniformly weighted

        if self.weighted:
            if any(distances == 0):
                weights[distances != 0] = 0  # if some neighbours have distance 0, their weight is 1, and others' is 0
            else:
                weights /= distances  # each weight equal 1/distance (the shorter the distance, the bigger the weight)

        return weights


class KNN_Classifier(KNN):

    def _vote(self, classes, distances):

        weights = self._get_weights(distances)

        prediction = None
        max_weighted_frequency = 0

        for c in classes:

            weighted_frequency = np.sum(weights[classes == c])

            if weighted_frequency > max_weighted_frequency:

                prediction = c
                max_weighted_frequency = weighted_frequency

        return prediction


class KNN_Regressor(KNN):

    def _vote(self, targets, distances):

        weights = self._get_weights(distances)

        return np.sum(weights * targets) / np.sum(weights)
