import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from supervised.logistic_regression import LogisticRegression
from supervised.linear_regression import AnalyticalLinearRegression
from utils.preprocessing import add_dummy_feature

seed = 2
np.random.seed(seed)

n_samples = 8

X = np.array([i for i in np.linspace(-1, 0, n_samples // 2)] + [i for i in np.linspace(0.5, 1.5, n_samples // 2)])
X_ = np.array([i for i in np.linspace(-1, 0, n_samples // 2)] +
              [i for i in np.linspace(0.5, 1.5, n_samples // 2)] + [3, 4, 5])

y = np.zeros(n_samples)
y_ = np.zeros(len(X_))

y[n_samples // 2:] = 1
y_[n_samples // 2:] = 1

reg1 = AnalyticalLinearRegression()
reg2 = AnalyticalLinearRegression()

reg1.fit(X[:, None], y)
reg2.fit(X_[:, None], y_)


fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(211)
ax1.set_ylim([-0.2, 1.2])

class1, = ax1.plot(X[n_samples // 2:], y[n_samples // 2:], "xg")
class2, = ax1.plot(X[:n_samples // 2], y[:n_samples // 2], "xr")

decision_boundary = (0.5 - reg1.coef_[0]) / reg1.coef_[1]
decision, = ax1.plot([decision_boundary, decision_boundary], [-0.5, 1.5], '--')
ax1.plot(X, reg1.predict(X[:, None]), "b")

ax1.legend([decision, class1, class2], ["decision boundary", "Class 1", "Class 2"])

ax2 = fig.add_subplot(212)
ax2.set_ylim([-0.2, 1.2])

class1, = ax2.plot(X_[n_samples // 2:], y_[n_samples // 2:], "xg")
class2, = ax2.plot(X_[:n_samples // 2], y_[:n_samples // 2], "xr")
decision_boundary = (0.5 - reg2.coef_[0]) / reg2.coef_[1]
decision, = ax2.plot([decision_boundary, decision_boundary], [-0.5, 1.5], '--')

ax2.plot(X_, reg2.predict(X_[:, None]), "b")
ax2.legend([decision, class1, class2], ["decision boundary", "Class 1", "Class 2"])

plt.show()
