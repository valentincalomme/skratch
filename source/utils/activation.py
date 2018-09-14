import numpy as np


class Sigmoid():

    def __call__(self, x):

        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x):

        p = self.__class__(x)

        return p * (1.0 - p)


class Softmax():

    def __call__(self, x):

        num = np.exp(x - np.max(x, axis=-1, keepdims=True))
        den = np.sum(num, axis=-1, keepdims=True)

        return num / den

    def gradient(self, x):

        p = self.__call__(x)

        return p * (1.0 - p)


class SoftPlus():

    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))


class ReLU():

    def __call__(self, x):
        return np.where(x >= 0.0, x, 0.0)

    def gradient(self, x):
        return np.where(x >= 0.0, 1.0, 0.0)


class TanH():

    def __call__(self, x):
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

    def gradient(self, x):
        return 1.0 - np.power(self.__call__(x), 2.0)


class ELU():

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1.0))

    def gradient(self, x):
        return np.where(x >= 0.0, 1.0, self.__call__(x) + self.alpha)


class LeakyReLU():

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0.0, 1.0, self.alpha)
