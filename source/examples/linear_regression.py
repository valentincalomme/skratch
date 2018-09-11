import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from supervised.linear_regression import AnalyticalLinearRegression, LinearRegression

from utils.preprocessing import add_dummy_feature, polynomial_features, MinMaxScaler
from utils.optimization import numerical_gradient, StochasticGradientDescentOptimizer
from utils.regularization import LASSO, Ridge, ElasticNet

seed = 2
np.random.seed(seed)

MIN = -1
MAX = 1
n_samples = 50
n_noisy_samples = 0
degree = 5

X = np.linspace(MIN, MAX, n_samples)
X = np.array([[x ** i for i in range(1, degree + 1)] for x in X.squeeze()])

weights = np.random.normal(0, 2, degree+1)

noise = np.zeros(n_samples)
noisy_instances = np.random.choice(np.arange(n_samples), n_noisy_samples)
noise[noisy_instances] = 3*np.sin(np.linspace(MIN, MAX, n_noisy_samples))

y = np.linspace(MIN, MAX, n_samples)
y += np.random.normal(0, 0.1, n_samples)

y += noise
y[0] += 2
y[-1] -= 2


features = add_dummy_feature(X)
fit_intercept = True

reg = LinearRegression(tol=1E-6,
                       optimizer=StochasticGradientDescentOptimizer(learning_rate=0.002),
                       regularizer=LASSO(_lambda=5), fit_intercept=fit_intercept, seed=seed)




#######################################################
###################### Animation ######################
#######################################################

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ims = []

for weights_, new_loss in reg._fit(X, y):

    lines = []

    y_pred = features.dot(weights_)  # reg.predict(X, weights_)

    correct, = ax.plot(X[:, 0].squeeze(), y, '.r')
    prediction, = ax.plot(X[:, 0].squeeze(), y_pred, 'b')

    ax.legend([correct, prediction], ["correct", "prediction"])
    lines.append(prediction)
    lines.append(correct)

    ims.append(lines)

anim = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=50)

plt.show()
