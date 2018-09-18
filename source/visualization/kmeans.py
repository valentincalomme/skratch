import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from unsupervised.kmeans import KMeans
from datasets.clustering import blobs, noisy_moons_

n_samples = 1000

seed = 7

np.random.seed(seed)

X = blobs(n_samples=n_samples, cluster_std= 0.5, n_features=2, centers=3, random_state=seed)
# X, _ = noisy_moons_
# X = np.random.rand(n_samples, 2)

kmeans = KMeans(k=3)
colors = np.random.rand(kmeans.k, 3)
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

ims = []

for c, l in kmeans._fit(X):

    lines = []

    for i in range(kmeans.k):

        line, = ax.plot(X[l == i][:, 0], X[l == i][:, 1], '.', c=colors[i], ms=30)

    line, = ax.plot(c[:, 0], c[:, 1], 'o', c='black', ms=30)
    ims.append(ax.lines[-(kmeans.k + 1):])

anim = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=100)
# anim.save("kmeans.mp4")
plt.show()
