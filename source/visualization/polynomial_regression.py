import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from supervised.linear_regression import LinearRegression
from utils.preprocessing import polynomial_features
from utils.regularization import BaseRegularizer
from utils.optimization import StochasticGradientDescentOptimizer

seed = 20

np.random.seed(seed)

n_samples = 50

X = np.reshape(np.linspace(-1, 1, n_samples), (n_samples, 1))

X1 = np.array([[x ** i for i in range(1, 1 + 1)] for x in X.squeeze()])  # polynomial_features(X, 1)
X2 = np.array([[x ** i for i in range(1, 2 + 1)] for x in X.squeeze()])  # polynomial_features(X, 2)
X3 = np.array([[x ** i for i in range(1, 3 + 1)] for x in X.squeeze()])  # polynomial_features(X, 3)
X4 = np.array([[x ** i for i in range(1, 4 + 1)] for x in X.squeeze()])  # polynomial_features(X, 4)

weights1 = np.random.normal(1, 2, 1)
y1 = (X1 + np.random.normal(0, 0.1, size=X1.shape)).dot(weights1)  # function of the curve with some normal noise added

weights2 = np.random.normal(1, 2, 2)
y2 = (X2 + np.random.normal(0, 0.1, size=X2.shape)).dot(weights2)  # function of the curve with some normal noise added

weights3 = np.random.normal(1, 2, 3)
y3 = (X3 + np.random.normal(0, 0.1, size=X3.shape)).dot(weights3)  # function of the curve with some normal noise added

weights4 = np.random.normal(1, 2, 4)
y4 = (X4 + np.random.normal(0, 0.1, size=X4.shape)).dot(weights4)  # function of the curve with some normal noise added

reg1 = LinearRegression(fit_intercept=False, optimizer=StochasticGradientDescentOptimizer(),
                        regularizer=BaseRegularizer())
reg2 = LinearRegression(fit_intercept=False, optimizer=StochasticGradientDescentOptimizer(),
                        regularizer=BaseRegularizer())
reg3 = LinearRegression(fit_intercept=False, optimizer=StochasticGradientDescentOptimizer(),
                        regularizer=BaseRegularizer())
reg4 = LinearRegression(fit_intercept=False, optimizer=StochasticGradientDescentOptimizer(),
                        regularizer=BaseRegularizer())


fig1 = plt.figure(figsize=(5, 5))
fig2 = plt.figure(figsize=(5, 5))
fig3 = plt.figure(figsize=(5, 5))
fig4 = plt.figure(figsize=(5, 5))

ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)
ax3 = fig3.add_subplot(1, 1, 1)
ax4 = fig4.add_subplot(1, 1, 1)


ys = [y1, y2, y3, y4]
Xs = [X1, X2, X3, X4]
regs = [reg1, reg2, reg3, reg4]
axs = [ax1, ax2, ax3, ax4]
figs = [fig1, fig2, fig3, fig4]
anims = []

for degree in range(1, 5):

    ims = []

    fig = figs[degree - 1]
    reg = regs[degree - 1]
    X = Xs[degree - 1]
    y = ys[degree - 1]

    ax = axs[degree - 1]

    ims = []

    X = polynomial_features(X, degree)

    reg.fit_intercept = False
    reg.coef_ = reg._initialize_weights(X)

    for _weights, _loss in reg._fit(X, y):
        lines = []
        y_ = reg.predict(X, _weights)
        correct, = ax.plot(X[:, 0].squeeze(), y, '.r')
        prediction, = ax.plot(X[:, 0].squeeze(), y_, 'b')

        lines.append(prediction)
        lines.append(correct)

        ax.legend([prediction, correct], ["Fitted Line", "Data"])

        ims.append(lines)

    anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=50)
    anims.append(anim)
    anim.save(f"visualization/gif/polymial_regression_{degree}.mp4")

plt.show()
