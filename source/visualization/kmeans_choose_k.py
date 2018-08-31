import numpy as np
import matplotlib.pyplot as plt


from unsupervised.kmeans import KMeans

n_samples = 100
n_features = 2

X = np.random.rand(n_samples, n_features)

inertias = []

for k in range(1, 10):  # len(X)):

    kmeans = KMeans(k=k)
    kmeans.fit(X)

    print(kmeans.inertia_)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 20))
plt.plot(range(1, len(inertias) + 1), inertias)
plt.show()
