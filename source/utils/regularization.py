import numpy as np

class Ridge(object):
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, theta):
        """
        Computes the cost based on weights theta
        :param theta: weights
        :return: cost of using ridge regression
        """
        return self._lambda * 0.5 * np.sum(np.square(theta))

    def gradient(self, theta):
        """
        Computes the gradient of weights theta using ridge regularization
        :param theta: weights
        :return: gradient for each weight
        """
        return self._lambda * theta


class LASSO(object):
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, theta):
        """
        Computes the cost based on weights theta
        :param theta: weights
        :return: cost of using LASSO regression
        """
        return self._lambda * np.sum(np.abs(theta))

    def gradient(self, theta):
        """
        Computes the gradient of weights theta using LASSO regularization
        :param theta: weights
        :return: gradient for each weight
        """
        return self._lambda * np.sign(theta)


class ElasticNet(object):
    def __init__(self, _lambda, l1_ratio=0.5):
        self.ratio = l1_ratio
        self._lambda = _lambda

    def __call__(self, theta):
        l1 = LASSO(_lambda = 1)
        l2 = Ridge(_lambda = 1)

        return self._lambda * (self.ratio * l1(theta) + (1 - self.ratio) * l2(theta))

    def gradient(self, theta):
        l1 = LASSO(_lambda = 1)
        l2 = Ridge(_lambda = 1)
        
        return self._lambda * (self.ratio * l1.gradient(theta) + (1 - self.ratio) * l2.gradient(theta))
