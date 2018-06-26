import numpy as np
from sklearn import datasets


def blobs(n_samples=100,
          n_features=2,
          centers=3,
          cluster_std=1.0,
          center_box=(-10.0, 10.0),
          shuffle=True,
          random_state=None):

    return datasets.make_blobs(n_samples=n_samples,
                               n_features=n_features,
                               centers=centers,
                               cluster_std=cluster_std,
                               center_box=center_box,
                               shuffle=shuffle,
                               random_state=random_state)[0]


def circles(n_samples=100,
            shuffle=True,
            noise=None,
            random_state=None,
            factor=0.8):

    return datasets.make_circles(n_samples=n_samples,
                                 shuffle=shuffle,
                                 noise=noise,
                                 random_state=random_state,
                                 factor=factor)[0]


def moons(n_samples=100,
          shuffle=True,
          noise=None,
          random_state=None,
          ):

    return datasets.make_moons(n_samples=n_samples,
                               shuffle=shuffle,
                               noise=noise,
                               random_state=random_state)[0]


# Examples from http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

n_samples = 150

noisy_circles_ = datasets.make_circles(n_samples=n_samples, factor=.5,
                                       noise=.05)

noisy_moons_ = datasets.make_moons(n_samples=n_samples, noise=.05)

blobs_ = datasets.make_blobs(n_samples=n_samples, random_state=8)

no_structure_ = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170

X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

transformation = [[0.6, -0.6], [-0.4, 0.8]]

X_aniso = np.dot(X, transformation)

aniso_ = (X_aniso, y)

# blobs with varied variances
varied_ = datasets.make_blobs(n_samples=n_samples,
                              cluster_std=[1.0, 2.5, 0.5],
                              random_state=random_state)
