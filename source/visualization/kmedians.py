import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from unsupervised.kmedians import KMedians
from datasets.clustering import blobs, noisy_moons_

n_samples = 2000
k = 4

# X = blobs(n_samples=n_samples, n_features=2, centers=3)
# X, _ = noisy_moons_
X = np.random.rand(n_samples, 2)


kmedians = KMedians(k=k)
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
anim.save("visualization/gif/kmedians.mp4")
plt.show()
