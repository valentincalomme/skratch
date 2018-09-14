import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from supervised.linear_regression import LinearRegression
from utils.regularization import LASSO
from utils.optimization import StochasticGradientDescentOptimizer
from utils.preprocessing import add_dummy_feature, polynomial_features


seed = 2
np.random.seed(seed)

MIN = -1
MAX = 1
n_samples = 50
n_noisy_samples = 0
degree = 5

reg = LinearRegression(tol=1E-4, seed=seed)

X = np.linspace(MIN, MAX, n_samples)
X = np.array([[x ** i for i in range(1, degree + 1)] for x in X.squeeze()])

weights = np.random.normal(0, 2, degree + 1)
# weights[np.random.choice(np.arange(len(weights)), len(weights)//2)] = 0

noise = np.zeros(n_samples)
noisy_instances = np.random.choice(np.arange(n_samples), n_noisy_samples)
noise[noisy_instances] = 3 * np.sin(np.linspace(MIN, MAX, n_noisy_samples))

y = np.linspace(MIN, MAX, n_samples)
y += np.random.normal(0, 0.1, n_samples)

y += noise
y[0] += 2
y[-1] -= 2

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_ylim([-2, 2])


# for _lambda in [0, 1, 0.1, 0.5, 0.01, 0.001]:

#     X_ = np.linspace(1.5 * MIN, 1.5 * MAX, n_samples)
#     X_ = np.array([[x ** i for i in range(1, degree + 1)] for x in X_.squeeze()])

#     reg = LinearRegression(tol=1E-4, seed=seed, regularizer=LASSO(_lambda=_lambda))

#     reg.fit(X, y)
#     y_ = reg.predict(X_)

#     not_noisy = np.array([i for i in range(n_samples) if i not in noisy_instances])

#     correct, = ax.plot(X[not_noisy, 0].squeeze(), y[not_noisy], '.g')
#     prediction, = ax.plot(X_[:, 0].squeeze(), y_, 'b')

#     ax.legend([correct, prediction], ["correct", f"prediction (lambda = {_lambda})"])

ims = []

from sklearn.linear_model import ElasticNet

for _lambda in np.linspace(0, 1, 100):

    X_ = np.linspace(1.5 * MIN, 1.5 * MAX, n_samples)
    X_ = np.array([[x ** i for i in range(1, degree + 1)] for x in X_.squeeze()])

    lines = []

    # reg = LinearRegression(tol=1E-6, seed=seed, optimizer=StochasticGradientDescentOptimizer(), regularizer=LASSO(_lambda=_lambda))

    reg = ElasticNet(alpha=0.03, l1_ratio=_lambda)

    reg.fit(X, y)
    print(reg.coef_)
    y_ = reg.predict(X_)

    not_noisy = np.array([i for i in range(n_samples) if i not in noisy_instances])

    correct, = ax.plot(X[not_noisy, 0].squeeze(), y[not_noisy], '.g')
    # noise, = ax.plot(X[noisy_instances,0].squeeze(), y[noisy_instances], 'Xr')
    prediction, = ax.plot(X_[:, 0].squeeze(), y_, 'b')

    ax.legend([correct, prediction], ["correct", "prediction"])
    lines.append(prediction)
    lines.append(correct)

    ims.append(lines)

anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=50)
anim.save(f"visualization/gif/elasticnet.mp4")
plt.show()
