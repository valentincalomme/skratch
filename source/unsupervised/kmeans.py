import numpy as np
import copy

from utils.distances import euclidean


class KMeans:

    def __init__(self, k=3, seed=0, n_runs=10):

        self.k = k
        self.rnd = np.random.RandomState(seed)
        self.n_runs = n_runs

    def predict(self, X):

        return self._compute_labels(X, self.centroids_)

    def fit(self, X, y=None):

        self.score_ = float('inf')  # initialize as worst possible performance/score

        for run in range(self.n_runs):

            for c, l in self._fit(X, y):
                centroids, labels = c, l

            performance = self._performance(X, centroids, labels)

            if performance < self.score_:

                self.score_ = performance
                self.centroids_ = centroids
                self.labels_ = labels

        return self

    def _fit(self, X, y=None):

        centroids = self._initialize_centroids(X)
        labels = self._compute_labels(X, centroids)
        old_labels = np.full_like(labels, -1)

        while any(old_labels != labels):

            old_labels = labels
            centroids = self._compute_centroids(X, labels)
            labels = self._compute_labels(X, centroids)

            yield centroids, labels

    def _initialize_centroids(self, X):

        self.rnd.shuffle(X.copy())
        return X[:self.k]

    def _distance(self, a, b):

        return euclidean(a, b)

    def _compute_centroids(self, X, labels):

        centroids = []

        for i in range(self.k):

            centroid = np.mean(X[labels == i], axis=0)
            centroids.append(centroid)

        return np.array(centroids)

    def _compute_labels(self, X, centroids):

        labels = []

        for x in X:

            distances = [self._distance(x, centroid) for centroid in centroids]
            label = np.argmin(distances)
            labels.append(label)

        return np.array(labels)

    def _performance(self, X, centroids, labels):

        distances = []

        for i, centroid in enumerate(centroids):

            distances.extend([self._distance(x, centroid) for x in X[labels == i]])

        return np.mean(np.power(distances, 2))

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from sklearn.datasets import make_blobs

    # X, _ = make_blobs(n_samples=1000, n_features=2, centers=5)
    X = np.random.rand(1000, 2)

    kmeans = KMeans(k=3)
    colors = np.random.rand(kmeans.k, 3)
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ims = []

    for c, l in kmeans._fit(X):

        lines = []

        for i in range(kmeans.k):

            line, = ax.plot(X[l == i][:, 0], X[l == i][:, 1], '.', c=colors[i])

        line, = ax.plot(c[:, 0], c[:, 1], 'X', c='black', ms=10)
        ims.append(ax.lines[-(kmeans.k + 1):])

    anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=100)
    plt.show()
