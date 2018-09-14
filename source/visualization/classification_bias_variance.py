import copy

import numpy as np
import matplotlib.pyplot as plt

from utils.activation import Sigmoid


sigmoid = Sigmoid()
n_samples = 500
n_features = 2
noise = 0.1
dim = 2
n_classes = 2
seed = 7


plt.figure(figsize=[20, 10])
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

y = f(X + np.random.normal(0, noise, size=(n_samples, n_features)))

h = 0.05

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = f(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

decision_boundary = lambda x: x * (-weights[0] / weights[1])

for i in range(n_classes + 1):
    plt.scatter(X[:, 0][y == i], X[:, 1][y == i], marker="o", s=40)

d = plt.contour(xx, yy, Z, [i for i in range(n_classes + 1)], colors=('black',), linewidths=3, linestyles='solid')


plt.show()

# plt.figure(figsize=(20,10))

# plt.subplot(3,1,1)

# data, = plt.plot(x, y, '.', markersize=10)
# good, = plt.plot(x, f(x), 'g')
# plt.legend([data, good, high_bias, high_variance], ["Noisy Data", "Denoised Data"])

# plt.subplot(3,1,2)

# data, = plt.plot(x, y, '.', markersize=10)
# high_bias, = plt.plot(x, np.zeros_like(x),  'b')
# plt.legend([data, good, high_bias, high_variance], ["Noisy Data", "High bias"])

# plt.subplot(3,1,3)

# data, = plt.plot(x, y, '.', markersize=10)
# high_variance, = plt.plot(x, y, 'r')
# plt.legend([data, good, high_bias, high_variance], ["Noisy Data", "High variance"])


# plt.show()
