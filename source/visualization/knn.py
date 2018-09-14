import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X = np.array([[0, 1], [1, 2], [0, 2], [1, 1.1], [1, 0.5], [0, 0.6], [0.2, 1.1], [1.1, 1.4],
              [5, 3], [4, 4], [3.7, 5], [3.1, 4], [4, 5.5], [5.1, 4.6], [3.8, 4.1], [4, 5.4]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

plt.figure(figsize=(20, 5))
plt.scatter(X[:, 0], X[:, 1],
            c=y,
            cmap=ListedColormap(['red', 'green']),
            s=100, marker='o')
plt.scatter([0.5], [0.5], c='gray', marker='X', s=200)
plt.show()
