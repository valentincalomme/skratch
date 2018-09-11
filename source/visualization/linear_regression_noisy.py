import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from supervised.linear_regression import LinearRegression
from utils.preprocessing import add_dummy_feature


seed = 6

n_samples = 10


reg = LinearRegression(tol=1E-6, seed=seed)

X = np.reshape(np.linspace(-1, 1, n_samples), (n_samples, 1))

np.random.seed(seed)
weights = np.random.normal(0, 2, degree+1)

noise = np.random.normal(0, 0.1, n_samples)
y = add_dummy_feature(X).dot(weights)+ noise

y[0] += np.random.normal(2, 3)


fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

ims = []

reg.coef_ = reg._initialize_weights(X)


for weights_, new_loss in reg._fit(X, y):

    lines = []

    y_ = add_dummy_feature(X).dot(weights_)

    prediction, = ax.plot(X.squeeze(), y, '.r')
    correct, = ax.plot(X.squeeze(), y_, 'b')
    
    lines.append(prediction)
    lines.append(correct)

    ims.append(lines)

anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=5)
anim.save("visualization/gif/linear_regression_noisy.mp4")
plt.show()