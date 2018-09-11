import numpy as np


class LinearKernel:

    def __init__(self, **kwargs):
        pass

    def __call__(self, x1, x2):
        return np.inner(x1, x2)


class PolynomialKernel:

    def __init__(self, power, coef, **kwargs):
        self.power = power
        self.coef = coef

    def __call__(self, x1, x2):
        return (np.inner(x1, x2) + self.coef)**self.power


class RBF:

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma

    def __call__(self, x1, x2):

        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-self.gamma * distance)
