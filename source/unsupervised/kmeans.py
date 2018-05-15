import numpy as np

from utils.distances import euclidean


class KMeans:

    def __init__(self, k=3, seed=0, n_runs=10):

        self.k = k
        self.seed = seed
        np.random.seed(self.seed)

        self.n_runs = n_runs

    def predict(self, X):

        pass

    def fit(self, X, y=None):

        self.score_ = float('inf')  # initialize as worst possible performance/score

        for run in range(self.n_runs):

            centroids = self._initialize_centroids(X)

            for labels_, centroids_ in self._update_centroids(X, centroids):

                


        self.centroids_ = np.array([[]])

        self.labels_ = np.array([])

        return self

    def _distance(self, a, b):

        return euclidean(a, b)

    def _update_centroids(self):
        pass

    def _initialize_centroids(self, X):

        pass

    def _evaluate_clustering(self):

        pass
