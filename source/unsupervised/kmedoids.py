import numpy as np
import copy

from unsupervised.kmeans import KMeans
from utils.distances import pdist, euclidean, manhattan, cosine


class KMedoids(KMeans):

    def __init__(self, k=3, seed=0, n_runs=10):

        super().__init__(k=k, seed=seed, n_runs=n_runs)

    def _distance(self, a, b):

        return manhattan(a, b)

    def _compute_centroids(self, X, labels):

        centroids = []

        for i in range(self.k):

            distances = pdist(X[labels == i], self._distance)
            within_cluster_sum_of_distances = np.sum(distances, axis=0)

            centroid = X[labels == i][np.argmin(within_cluster_sum_of_distances)]

            centroids.append(centroid)

        return np.array(centroids)

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from sklearn.datasets import make_blobs

    n_samples = 200

    X = []
    for i in np.linspace(0, 1, np.sqrt(n_samples), dtype=np.float):
        for j in np.linspace(0, 1, np.sqrt(n_samples), dtype=np.float):
            X.append([i, j])

    X = np.array(X)

    # X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=30)

    kmedoids = KMedoids(k=3)
    colors = np.random.rand(kmedoids.k, 3)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)

    ims = []

    for c, l in kmedoids._fit(X):

        lines = []

        for i in range(kmedoids.k):

            line, = ax.plot(X[l == i][:, 0], X[l == i][:, 1], '.', c=colors[i], ms=30)

        line, = ax.plot(c[:, 0], c[:, 1], 'o', c='black', ms=30)
        ims.append(ax.lines[-(kmedoids.k + 1):])

    anim = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=100)
    # anim.save("kmedoids.mp4")
    plt.show()
