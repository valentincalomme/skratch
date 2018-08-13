"""Linear Regression

https://www.cs.toronto.edu/~frossard/post/linear_regression/

"""
import itertools
import copy

import numpy as np

from utils.preprocessing import add_dummy_feature, StandardScaler
from utils.regularization import BaseRegularizer
from utils.optimization import numerical_gradient


class LinearRegression:

    def __init__(self, optimizer, fit_intercept=True, regularizer=None, tol=1E-8, seed=0):

        self.optimizer = optimizer
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.regularizer = BaseRegularizer() if regularizer is None else regularizer
        self.rnd = np.random.RandomState(seed)

    def predict(self, X, weights=None):

        X = add_dummy_feature(X) if self.fit_intercept else X
        weights = self.weights_ if weights is None else weights

        return X.dot(weights)

    def fit(self, X, y):

        self.weights_ = self._initialize_weights(X)

        for weights, loss in self._regression(X, y, self.weights_):
            self.weights_ = weights
            self.loss_ = loss

        return self

    def _initialize_weights(self, X, minimum=0, maximum=1):

        _, n_features = X.shape

        n_features += self.fit_intercept  # add a weight for the intercept if it is being used

        return (maximum - minimum) * self.rnd.rand(n_features) + minimum

    def _regression(self, X, y, weights):

        loss = self._loss(X, y)
        loss_gradient = numerical_gradient(loss)

        old_loss = float('inf')
        new_loss = loss(weights)

        while old_loss - new_loss > self.tol:

            yield weights, new_loss

            old_loss = new_loss
            weights = self.optimizer.update(weights, loss_gradient(weights))
            new_loss = loss(weights)

    def _loss(self, X, y):

        prediction_loss = lambda weights: np.mean((y - self.predict(X, weights)) ** 2) * 0.5
        regularization_loss = lambda weights: self.regularizer(weights)

        return lambda weights: prediction_loss(weights) + regularization_loss(weights)

# if __name__ == "__main__":

#     from utils.optimization import GradientDescentOptimizer
#     from utils.regularization import *

#     optimizer = GradientDescentOptimizer(learning_rate=0.005, momentum=0.1)

#     num_samples = 100
#     num_features = 5

#     reg = LinearRegression(optimizer, regularizer=Ridge(_lambda=0.0))

#     rnd = np.random.RandomState(10)

#     X = rnd.rand(num_samples, num_features)
#     weights = rnd.rand(X.shape[1] + 1)
#     y = add_dummy_feature(X).dot(weights)

#     reg.fit(X, y)
