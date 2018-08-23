"""Linear Regression

https://www.cs.toronto.edu/~frossard/post/linear_regression/

"""

import numpy as np

from supervised.linear_regression import LinearRegression
from utils.preprocessing import add_dummy_feature
from utils.activation import Sigmoid


class LogisticRegression(LinearRegression):

    def _predict(self, X, weights=None):

        X = add_dummy_feature(X) if self.fit_intercept else X
        weights = self.coef_ if weights is None else weights

        sigmoid = Sigmoid()

        return sigmoid(X.dot(weights))

    def predict(self, X, weights=None):

        return self._predict(X, weights) > 0.5

    def _loss_function(self, X, y):

        prediction_loss = lambda weights: np.mean((y - self._predict(X, weights)) ** 2) * 0.5
        regularization_loss = lambda weights: self.regularizer(weights)

        return lambda weights: prediction_loss(weights) + regularization_loss(weights)

    def _loss_gradient(self, X, y):

        features = add_dummy_feature(X) if self.fit_intercept is True else X

        prediction_loss_gradient = lambda weights: (self._predict(X, weights) - y).dot(features) / len(features)
        regularization_loss_gradient = lambda weights: self.regularizer.gradient(weights)

        return lambda weights: prediction_loss_gradient(weights) + regularization_loss_gradient(weights)
