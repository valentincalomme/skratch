import numpy as np


def numerical_gradient(f, wrt=None):

    def gradient(x, wrt):

        h = 1E-8

        gradients = np.zeros(len(x), dtype=float)

        if wrt is None:
            wrt = np.arange(len(x))

        for i in wrt:
            diff = np.zeros(len(x), dtype=float)
            diff[i] = h

            gradients[i] = (f(x + diff) - f(x - diff)) / (2 * np.sum(diff))

        return gradients

    return lambda x: gradient(x, wrt)


class StochasticGradientDescentOptimizer:

    def __init__(self, learning_rate=0.2, momentum=0):

        self.momentum = momentum
        self.learning_rate = learning_rate
        self.parameter_update = None

    def step(self, parameters, gradient, **kwargs):

        if self.parameter_update is None:
            self.parameter_update = np.zeros(len(parameters))

        self.parameter_update = self.momentum * self.parameter_update + (1 - self.momentum) * gradient(parameters)

        return parameters - self.learning_rate * self.parameter_update


class NesterovAcceleratedGradientOptimizer:

    def __init__(self, learning_rate=0.001, momentum=0.4):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameter_update = None

    def step(self, parameters, gradient, **kwargs):

        if self.parameter_update is None:
            self.parameter_update = np.zeros(len(parameters))

        approximate_future_gradient = np.clip(gradient(parameters - self.momentum * self.parameter_update), -1, 1)

        self.parameter_update = self.momentum * self.parameter_update + self.learning_rate * approximate_future_gradient

        return parameters - self.parameter_update