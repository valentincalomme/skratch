import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from unsupervised.kmedoids import KMedoids
from datasets.clustering import blobs, noisy_moons_
from utils.distances import pdist, euclidean, manhattan, cosine

n_samples = 1000
k = 4

# X = blobs(n_samples=n_samples, n_features=2, centers=3)
X, _ = noisy_moons_
# X = np.random.rand(n_samples, 2)

kmedoids = KMedoids(k=4)
colors = np.random.rand(kmedoids.k, 3)
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

ims = []

for c, l in kmedoids._fit(X):

    lines = []

    for i in range(kmedoids.k):

        line, = ax.plot(X[l == i][:, 0], X[l == i][:, 1], '.', c=colors[i], ms=30)

    line, = ax.plot(c[:, 0], c[:, 1], '.', c='black', ms=30)
    ims.append(ax.lines[-(kmedoids.k + 1):])

anim = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=100)
# anim.save("kmeans.mp4")
plt.show()
