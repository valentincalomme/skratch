import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.preprocessing import add_dummy_feature
from supervised.linear_regression import LinearRegression

seed = 1
np.random.seed(seed)

n_samples = 50


X = np.linspace(0, 1, n_samples).reshape(n_samples, 1)
weights = np.random.rand(1)
y = X.dot(weights) + np.random.normal(0, 0.15, n_samples)

from utils.regularization import BaseRegularizer
from utils.optimization import StochasticGradientDescentOptimizer

reg = LinearRegression(fit_intercept=False, optimizer=StochasticGradientDescentOptimizer(),
                       regularizer=BaseRegularizer())

loss = reg._loss_function(X, y)
loss_gradient = reg._loss_gradient(X, y)


potential_weights = np.linspace(-5, 5, 1000)

losses = [loss(np.array([w])) for w in potential_weights]
loss_gradients = [loss_gradient(np.array([w]))[0] for w in potential_weights]

reg.coef_ = np.array([-5])


fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(211)
ax1.set_xlim([0, 1])
ax1.set_ylim([-1, 1])
ax2 = fig.add_subplot(212)
ax2.set_xlim([-5, 5])
ax2.set_ylim([-0.5, 7])
ims = []

for _weights, _loss in reg._fit(X, y):

    lines1 = []
    lines2 = []

    y_ = reg.predict(X, _weights)

    correct, = ax1.plot(X.squeeze(), y, '.r')
    prediction, = ax1.plot(X.squeeze(), y_, 'b')

    lines1.append(prediction)
    lines1.append(correct)

    ax1.legend([prediction, correct], ["Fitted Line", "Data"])

    loss, = ax2.plot(potential_weights, losses, 'b')

    current_loss, = ax2.plot(_weights[0], _loss, 'Xr', markersize=10)
    ax2.legend([loss, current_loss], ["Loss", "Current Loss"])

    lines2.append(current_loss)
    lines2.append(loss)

    ims.append([*lines1, *lines2])


anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=100)

# anim.save("visualization/gif/gradient_descent.mp4")

plt.show()
