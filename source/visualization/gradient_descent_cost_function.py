import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing import add_dummy_feature
seed = 0
np.random.seed(seed)

n_samples = 10


X = np.linspace(0, 1, n_samples).reshape(n_samples, 1)
weights = np.random.rand(1)
y = X.dot(weights)


def _loss_function(X, y):

    prediction_loss = lambda weights: np.mean((y - X.dot(weights)) ** 2) * 0.5

    return lambda weights: prediction_loss(weights)


def _loss_gradient(X, y):

    features = X

    prediction_loss_gradient = lambda weights: (X.dot(weights) - y).dot(features) / len(features)

    return lambda weights: prediction_loss_gradient(weights)

loss = _loss_function(X, y)
loss_gradient = _loss_gradient(X, y)


potential_weights = np.linspace(-5, 5, 1000)

losses = [loss(np.array([w])) for w in potential_weights]
loss_gradients = [loss_gradient(np.array([w]))[0] for w in potential_weights]


plt.figure(figsize=(20, 10))

# plt.subplot(221)

# plt.plot(X.squeeze(), y)

# plt.subplot(222)

line, = plt.plot(potential_weights, losses, 'r')

plt.legend([line], ["Loss"])
plt.annotate(f"loss minimum at {weights[0]:.4f}",
             xy=(potential_weights[np.argmin(losses)], np.min(losses)),
             xytext=(potential_weights[np.argmin(losses)], np.min(losses) + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))


# plt.subplot(223)
# plt.plot(potential_weights, loss_gradients, 'b')

# plt.annotate(f"loss gradient is zero at {weights[0]:.4f}",
#              xy=(weights, 0),
#              xytext=(weights-0.5, 0.08),
#              arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
