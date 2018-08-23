import numpy as np

from utils.kernels import RBF, LinearKernel, PolynomialKernel
from utils.optimization import GradientDescentOptimizer


class GaussianProcessRegressor:

    def __init__(self, kernel=RBF(1.0), seed=0):

        self.kernel = kernel
        self.rnd = np.random.RandomState(seed)

    def fit(self, X, y):

        return self

    def predict(self, X):

        return None

    def predict_proba(self, X):

        return None
