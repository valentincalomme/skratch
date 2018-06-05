import numpy as np
import copy

from unsupervised.kmeans import KMeans
from utils.distances import manhattan


class KMedians(KMeans):

    def __init__(self, k=3, seed=0, n_runs=10):

        super().__init__(k=k, seed=seed, n_runs=n_runs)

    def _distance(self, a, b):

        return manhattan(a, b)

    def _compute_centroids(self, X, labels):

        centroids = []

        for i in range(self.k):

            centroid = np.array([np.median(dim) for dim in X[labels == i].T])
            centroids.append(centroid)

        return np.array(centroids)


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from sklearn.datasets import make_blobs

    n_samples = 1000

    X = []
    for i in np.linspace(0, 1, np.sqrt(n_samples), dtype=np.float):
        for j in np.linspace(0, 1, np.sqrt(n_samples), dtype=np.float):
            X.append([i, j])

    X = np.array(X)

    # X, _ = make_blobs(n_samples=100, n_features=2, centers=3)

    kmedians = KMedians(k=2)
    colors = np.random.rand(kmedians.k, 3)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)

    ims = []

    for c, l in kmedians._fit(X):

        lines = []

        for i in range(kmedians.k):

            line, = ax.plot(X[l == i][:, 0], X[l == i][:, 1], '.', c=colors[i], ms=30)

        line, = ax.plot(c[:, 0], c[:, 1], 'o', c='black', ms=30)
        ims.append(ax.lines[-(kmedians.k + 1):])

    anim = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=100)
   # anim.save("KMedians.mp4")
    plt.show()
