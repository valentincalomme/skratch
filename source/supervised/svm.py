"""Implementation coming soon"""


# """Support vector machine"""

# import numpy as np

# from utils.kernels import RBF, LinearKernel, PolynomialKernel
# from utils.optimization import GradientDescentOptimizer
# from utils.distances import pdist


# class GaussianProcessRegressor:

#     def __init__(self, C=1.0, kernel=RBF(1.0), seed=0):

#         self.C = C
#         self.kernel = kernel
#         self.rnd = np.random.RandomState(seed)

#     def fit(self, X, y):

#         n_samples, n_features = X.shape

#         self.kernel_matrix = self._compute_kernel_matrix(X)

#         return self

#     def _compute_kernel_matrix(self, X):

#         return pdist(X, self.kernel)

#     def predict(self, X):

#         return None