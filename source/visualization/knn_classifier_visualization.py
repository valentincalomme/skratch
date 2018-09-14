import math
from math import pi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from supervised.knn import KNN_Classifier
from utils.activation import Sigmoid
from datasets.classification import blobs


def points(n=10):
    r = 1.5

    return [(math.cos(2 * pi / n * x) * r, math.sin(2 * pi / n * x) * r) for x in range(0, n + 1)]
    # return [(x, -np.sin(5*x)) for x in np.linspace(-1, 1, n + 1)]

seed = 1
n_points = 100
k = 3

np.random.seed(seed)
minimum = -2
maximum = 2
sigmoid = Sigmoid()
weights = np.random.rand(2)

# X_ = (maximum - minimum) * np.random.rand(n_points, 2) + minimum
# y = sigmoid(X_.dot(weights)) > 0.5
# y = np.random.randint(0, 2, n_points)
X_, y = blobs(n_points, centers=2, center_box=(0, 0), cluster_std=1)

fig = plt.figure(figsize=(40, 20))

colors = ['g', 'r']

ims = []

knn = KNN_Classifier(k=k)
knn.fit(X_, y)

for x0, x1 in points(n=500):

    X = np.array([[x0, x1]])
    neighbours_id, _ = knn._get_neighbours(X[0])
    neighbours = X_[neighbours_id]
    neighbours_y = y[neighbours_id]

    lines = []
    ax = fig.add_subplot(111)

    for neighbour, neighbour_y in zip(neighbours, neighbours_y):
        line, = ax.plot([x0, neighbour[0]], [x1, neighbour[1]], color=colors[neighbour_y])
        lines.append(line)

    line, = ax.plot(x0, x1, 'o', c=colors[knn.predict(X)[0]], ms=20)
    lines.append(line)

    for y_ in set(y):
        line, = ax.plot(X_[y == y_][:, 0], X_[y == y_][:, 1], 'o', color=colors[y_], ms=10)
        lines.append(line)

    ims.append(lines)

    # print((ims))

anim = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=0)

plt.show()

anim.save("visualization/gif/knn_classification_anim.mp4", frame_size=(20, 10))
