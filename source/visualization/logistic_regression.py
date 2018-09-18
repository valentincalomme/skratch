import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from supervised.logistic_regression import LogisticRegression

from utils.preprocessing import add_dummy_feature, polynomial_features, MinMaxScaler
from utils.optimization import numerical_gradient, StochasticGradientDescentOptimizer
from utils.activation import Sigmoid

seed = 42
np.random.seed(seed)
sigmoid = Sigmoid()
degree = 1

n_samples = 300
n_features = 2

X = np.random.normal(0, 1, size=(n_samples, 2))
features = polynomial_features(X, degree)

weights = np.random.normal(0, 1, features.shape[1])

y = sigmoid(features.dot(weights)) > 0.5

clf = LogisticRegression(tol=1E-4, fit_intercept=False, seed=seed)

#######################################################
###################### Animation ######################
#######################################################

fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(111)
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 3])

ims = []
h = 0.01

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

XX = polynomial_features(np.c_[xx.ravel(), yy.ravel()], degree)

for weights_, new_loss in clf._fit(features, y):

    print(new_loss)
    lines = []

    Z = clf.predict(XX, weights_) > 0.5
    Z = Z.reshape(xx.shape)

    class1, = ax.plot(X[y == 0][:, 0], X[y == 0][:, 1], ".g", markersize=15)
    class2, = ax.plot(X[y == 1][:, 0], X[y == 1][:, 1], ".r", markersize=15)
    d = ax.contour(xx, yy, Z, [0.5], colors=('black',), linewidths=2, linestyles='solid')

    ax.legend([class1, class2], ["Class 1", "Class 2"])

    lines.append(class1)
    lines.append(class2)
    lines.append(d.collections[0])

    ims.append(lines)

anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=50)
# anim.save("visualization/gif/logistic_regression2.mp4")
plt.show()
