import numpy as np
import matplotlib.pyplot as plt
from datasets.clustering import blobs, moons, circles

from sklearn.cluster import DBSCAN

num_points = 1000

seed = 2

X1 = blobs(n_samples=num_points, centers=3, cluster_std=.4, random_state=seed)
X2 = moons(n_samples=num_points//10, random_state=seed)
X3 = circles(n_samples=num_points, factor=.5, noise=.05)
X4 = blobs(n_samples=num_points, cluster_std=1, centers=2, random_state=seed)


plt.figure(figsize=(20, 10))

for clst, X, s in zip([DBSCAN(eps=1, min_samples=20),
                       DBSCAN(eps=.3),
                       DBSCAN(eps=.2, min_samples=10),
                       DBSCAN(eps=2)], [X1, X2, X3, X4], [111]):#[221, 222, 223, 224]):

    plots = []
    y = clst.fit_predict(X)

    plt.subplot(s)

    for i in range(len(set(y))):
        p = plt.scatter(X[:, 0][y == i], X[:, 1][y == i], marker="o", s=40)
        plots.append(p)

    # fitted = plt.scatter(X[:,0], X[:,1], y, 'b', markersize=10)
    plt.legend(plots, [f"Cluster {i+1}" for i in range(len(plots))])

plt.show()
