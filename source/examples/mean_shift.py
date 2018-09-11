import numpy as np
import matplotlib.pyplot as plt

from unsupervised.mean_shift import MeanShift

seed = 0

np.random.seed(seed)

cluster = MeanShift(bandwith=1)


n_samples = 10
X = np.random.normal(0, 2, (n_samples, 2))

for labels, centers in cluster._fit(X):

    print(labels, centers)

plt.plot(X[:, 0], X[:, 1], "xb")
plt.plot(centers[:, 0], centers[:, 1], "or")
plt.show()