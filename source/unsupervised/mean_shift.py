"""Mean shift"""

import numpy as np

from utils.kernels import RBF


class MeanShift:

    def __init__(self, kernel=RBF, bandwith=1, cluster_all=True):

        self.bandwith = bandwith
        self.cluster_all = cluster_all
        self.kernel = RBF(gamma=self.bandwith)

    def fit(self, X):

        for labels, centers in self._fit(X):

            self.labels_ = labels
            self.cluster_centers_ = centers

        return self

    def _fit(self, X):

        old_centers = None
        new_centers = X
        labels = -np.ones(len(X))

        while not self._has_converged(old_centers, new_centers):

            yield labels, new_centers

            old_centers = new_centers
            new_centers = []

            for i, center in enumerate(old_centers):

                shifted_center = self._shift(center, X)

                new_centers.append(shifted_center)

            new_centers = np.unique(new_centers, axis=0)

            labels = self._compute_labels(X, new_centers)

    def _has_converged(self, old_centers, new_centers):

        if old_centers is None or new_centers is None:
            return False
        else:
            return set(tuple(_) for _ in old_centers) == set(tuple(_) for _ in new_centers)

    def _shift(self, x, X):

        distances = np.array([self.kernel(x, x_) for x_ in X])
        # print(distances < self.bandwith)
        neighbourhood, = np.where(distances < self.bandwith)

        shifted_center = np.mean(X[neighbourhood], axis=0)

        return shifted_center

    def _compute_labels(self, X, centers):

        labels = []

        for x in X:

            distances = [self.kernel(x, center) for center in centers]
            label = np.argmin(distances)
            labels.append(label)

        return np.array(labels)

    def predict(self, X):

        labels = self._compute_labels(X, self.cluster_centers_)

        return labels

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # from unsupervised.mean_shift import MeanShift

    seed = 0

    np.random.seed(seed)

    cluster = MeanShift(bandwith=0.2)

    n_samples = 10
    X = np.random.normal(0, 2, (n_samples, 2))

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ims = []

    for labels, centers in cluster._fit(X):

        print("labels =", labels)
        print("centers =", centers)

        a, = ax.plot(X[:, 0], X[:, 1], "xb")
        b, = ax.plot(centers[:, 0], centers[:, 1], "or")

        ims.append([a, b])

    anim = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=100)

    plt.show()
