"""Linear Regression

https://www.cs.toronto.edu/~frossard/post/linear_regression/

"""

import numpy as np

from supervised.linear_regression import LinearRegression
from utils.preprocessing import add_dummy_feature
from utils.activation import Sigmoid


class LogisticRegression(LinearRegression):

    def predict(self, X, weights=None):

        if self.fit_intercept is True:
            X = add_dummy_feature(X)

        if weights is None:
            weights = self.coef_

        sigmoid = Sigmoid()

        return sigmoid(X.dot(weights))

    def predict_classes(self, X, weights=None):

        return self.predict(X, weights) > 0.5