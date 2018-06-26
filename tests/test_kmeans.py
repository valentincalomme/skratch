import itertools

import pytest
import numpy as np

from sklearn.cluster import KMeans as sklearn_KMeans
from unsupervised.kmeans import KMeans as skratch_KMeans

from datasets.clustering import noisy_moons_, noisy_circles_, aniso_, varied_, no_structure_, blobs_, blobs
from utils.evaluation import adjusted_rand_score

EPSILON = 0.1


# DATASETS = [noisy_moons_,
#             noisy_circles_,
#             aniso_,
#             varied_,
#             no_structure_,
#             blobs_]

DATASETS = [np.random.rand(100, n_feature) for n_feature in range(2, 6)]
KS = range(2, 8)


@pytest.mark.parametrize("k", KS)
def test_kmeans_vs_sklearn(k):

    n_samples = 100

    dataset = blobs(centers=k, n_samples=n_samples)

    skr = skratch_KMeans(k=k, n_runs=30, seed=None)
    skl = sklearn_KMeans(n_clusters=k, n_init=30)

    X = skr.fit(dataset).predict(dataset)
    Y = skl.fit(dataset).predict(dataset)

    assert adjusted_rand_score(X, Y) > 1 - EPSILON


@pytest.mark.parametrize("k, seed", itertools.product(KS, range(4)))
def test_kmeans_is_not_stochastic(k, seed):

    n_samples = 100

    dataset = blobs(centers=k, n_samples=n_samples)

    skr1 = skratch_KMeans(k=k, n_runs=30, seed=seed)
    skr2 = skratch_KMeans(k=k, n_runs=30, seed=seed)

    X = skr1.fit(dataset).predict(dataset)
    Y = skr2.fit(dataset).predict(dataset)

    assert adjusted_rand_score(X, Y) > 1 - EPSILON
