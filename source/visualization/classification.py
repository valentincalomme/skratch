import copy

import numpy as np
import matplotlib.pyplot as plt

from utils.activation import Sigmoid
from sklearn.svm import SVC


sigmoid = Sigmoid()
n_samples = 500
n_features = 2
noise = 0
dim = 2
n_classes = 2


#seed = 7

clf = SVC(C = 999 )


plt.figure(figsize=[20, 10])

for seed, s in zip([7, 17, 20, 19],  [221, 222, 223, 224]):

    plots = []
    plt.subplot(s)
    np.random.seed(seed)

    X = 2.0 * np.random.rand(n_samples, n_features) - 1.0
    weights = 2.0 * np.random.rand(n_features * dim) - 1.0


    def f(x):

        x = copy.copy(x)
        x = np.concatenate(tuple([x**i for i in range(1, dim + 1)]), axis=1)
        output = sigmoid(x.dot(weights))
        Y = np.zeros(len(x))

        for i in np.linspace(output.min(), output.max(), n_classes + 1):
            Y[output >= i] += 1

        return Y

    y = f(X + np.random.normal(0, noise, size=X.shape))


    clf.fit(X, y)

    #np.random.normal(0, noise, size=x.shape)
    h = 0.001

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])#f(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    for i in range(n_classes + 1):
        p = plt.scatter(X[:, 0][y == i], X[:, 1][y == i], marker="o", s=40)
        plots.append(p)

    d = plt.contour(xx, yy, Z, [i for i in range(n_classes + 1)], colors=('black',), linewidths=3, linestyles='solid')
    plt.legend(plots[1:], [f"Class {i}" for i in range(1, len(plots))])


plt.show()
