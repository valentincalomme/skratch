"""Linear Regression

https://www.cs.toronto.edu/~frossard/post/linear_regression/

"""
import numpy as np

from utils.preprocessing import add_dummy_feature
from utils.regularization import BaseRegularizer
from utils.optimization import StochasticGradientDescentOptimizer


class LinearRegression:

    def __init__(self,
                 optimizer=None,
                 regularizer=None,
                 fit_intercept=True,
                 tol=1E-4,
                 seed=0):

        self.optimizer = StochasticGradientDescentOptimizer() if optimizer is None else optimizer
        self.regularizer = BaseRegularizer() if regularizer is None else regularizer
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.rnd = np.random.RandomState(seed)

    def predict(self, X, weights=None):

        if self.fit_intercept:
            X = add_dummy_feature(X)

        if weights is None:
            weights = self.coef_

        return X.dot(weights)

    def fit(self, X, y):

        for w, l in self._fit(X, y):

            self.coef_ = w
            self.loss_ = l

        return self

    def _fit(self, X, y):

        weights = self._initialize_weights(X)

        loss_function = self._loss_function(X, y)
        loss_gradient = self._loss_gradient(X, y)

        old_loss = float('inf')
        new_loss = loss_function(weights)

        while old_loss - new_loss > self.tol:

            yield weights, new_loss

            old_loss = new_loss
            weights = self.optimizer.step(parameters=weights, gradient=loss_gradient)
            new_loss = loss_function(weights)

    def _initialize_weights(self, X, minimum=-1, maximum=1):

        _, n_features = X.shape

        n_features += self.fit_intercept  # add a weight for the intercept if it is being used

        return (maximum - minimum) * self.rnd.rand(n_features) + minimum

    def _loss_function(self, X, y):

        prediction_loss = lambda weights: 0.5 * (y - self.predict(X, weights)) ** 2
        regularization_loss = lambda weights: self.regularizer(weights)

        return lambda weights: np.mean(prediction_loss(weights) + regularization_loss(weights))

    def _loss_gradient(self, X, y):

        features = X
        if self.fit_intercept:
            features = add_dummy_feature(features)

        prediction_loss_gradient = lambda weights: (self.predict(X, weights) - y).dot(features) / len(features)
        regularization_loss_gradient = lambda weights: self.regularizer.gradient(weights)

        return lambda weights: prediction_loss_gradient(weights) + regularization_loss_gradient(weights)


class AnalyticalLinearRegression:

    def __init__(self, fit_intercept=True):

        self.fit_intercept = fit_intercept

    def predict(self, X, weights=None):

        if self.fit_intercept:
            X = add_dummy_feature(X)

        if weights is None:
            weights = self.coef_

        return X.dot(weights)

    def fit(self, X, y):

        if self.fit_intercept:
            X = add_dummy_feature(X)

        self.coef_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        return self.coef_
