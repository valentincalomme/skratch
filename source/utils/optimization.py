import numpy as np


def numerical_gradient(f):

    def _gradient(x, wrt=None):

        h = 1E-8

        gradients = np.zeros(len(x), dtype=float)

        if wrt is None:
            wrt = np.arange(len(x))

        for i in wrt:
            diff = np.zeros(len(x), dtype=float)
            diff[i] = h

            gradients[i] = (f(x + diff) - f(x - diff)) / (2 * np.sum(diff))

        return gradients

    return _gradient


class GradientDescentOptimizer:

    def __init__(self, learning_rate=0.2, momentum=0):

        self.momentum = momentum
        self.learning_rate = learning_rate

        self.weight_update = None

    def update(self, x, grad):

        if self.weight_update is None:
            self.weight_update = np.zeros(len(x))

        self.weight_update = self.momentum * self.weight_update + (1 - self.momentum) * grad

        return x - self.learning_rate * self.weight_update
