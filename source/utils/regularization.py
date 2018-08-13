"""Regularization module providing utility functions for Ridge, LASSO, and ElasticNet regularization

Regularization is a technique used in optimization to add a certain constraint on parameters to avoid overfitting.

This module provides 3 classes:
- LASSO
- Ridge
- ElasticNet

Each of these classes needs to be initialized with an argument lambda defining the weight given to the regularization.
Each class overrides the __call__ method in order to compute the value of the regularization. To do so, one has to pass
an iterable of weights to it. Each class also provides a gradient function, which also take in an iterable of weights,
and then returns the gradient of the regularization.

"""


import numpy as np


class BaseRegularizer:

    def __call__(self, theta):
        return 0

    def gradient(self, theta):
        return np.zeros_like(theta)


class Ridge(BaseRegularizer):

    def __init__(self, _lambda, **kwargs):
        self._lambda = _lambda

    def __call__(self, theta):

        return self._lambda * 0.5 * np.sum(theta**2)

    def gradient(self, theta):

        return self._lambda * theta


class LASSO(BaseRegularizer):

    def __init__(self, _lambda, **kwargs):
        self._lambda = _lambda

    def __call__(self, theta):

        return self._lambda * np.sum(np.abs(theta))

    def gradient(self, theta):

        return self._lambda * np.sign(theta)


class ElasticNet(BaseRegularizer):

    def __init__(self, _lambda, l1_ratio=0.5):
        self.ratio = l1_ratio
        self._lambda = _lambda

    def __call__(self, theta):

        l1 = LASSO(_lambda=1)
        l2 = Ridge(_lambda=1)

        return self._lambda * (self.ratio * l1(theta) + (1 - self.ratio) * l2(theta))

    def gradient(self, theta):

        l1 = LASSO(_lambda=1)
        l2 = Ridge(_lambda=1)

        return self._lambda * (self.ratio * l1.gradient(theta) + (1 - self.ratio) * l2.gradient(theta))
