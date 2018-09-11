import numpy as np
import matplotlib.pyplot as plt

from utils.activation import Sigmoid

n_samples = 20
x = np.linspace(-10, 10, n_samples)

sigmoid = Sigmoid()

plt.plot(x, sigmoid(x), "b")
plt.plot([0,0],[0,1],"--")
plt.show()
