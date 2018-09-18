import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from unsupervised.mean_shift import MeanShift
from datasets.clustering import blobs

seed = 1

np.random.seed(seed)

cluster = MeanShift(bandwith=1)

n_samples = 300
# np.random.normal(0, 1, (n_samples, 2))

X = blobs(n_samples=n_samples, random_state=seed)

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

ims = []

colors = np.random.rand(len(X) + 1, 3)

for labels, centers in cluster._fit(X):

    cluster.labels_ = labels
    cluster.cluster_centers_ = centers

    print("labels =", labels)
    print("centers =", centers)

    lines = []

    size = 50

    # labels = cluster.predict(X)

    for i in set(cluster.labels_):

        line, = ax.plot(X[labels == i][:, 0], X[labels == i][:, 1], '.', ms=size // 2, c="blue")#, c=colors[int(i + 1)])
        lines.append(line)

    b, = ax.plot(centers[:, 0], centers[:, 1], ".", c="black", ms=size)

    lines.append(b)
    ims.append(lines)

anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=100)
anim.save("visualization/gif/mean_shift.mp4")
plt.show()
