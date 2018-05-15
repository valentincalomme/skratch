import pytest
import itertools

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import scipy.spatial.distance as sc

import utils.distances as sk
from utils.evaluation import mse, accuracy
from utils.distances import euclidean
from supervised.knn import KNN_Classifier, KNN_Regressor

n_samples = 11
n_dims = 2

n_k = 10

k = np.random.choice(range(1, n_samples, 2), size=(n_k,))
weighted = [(False, "uniform"), (True, "distance")]

epsilon = 1E-2


distance_pairs = [
    dict(sc="cosine",         sk=sk.cosine),
    dict(sc="l2",             sk=sk.euclidean),
    dict(sc="chebyshev",      sk=sk.chebyshev),
    dict(sc="cityblock",      sk=sk.manhattan)
]


@pytest.mark.parametrize("k, weighted, pairs", itertools.product(k, weighted, distance_pairs))
def test_knn_classifier(k, weighted, pairs):

    X = np.random.rand(n_samples, n_dims)
    y = np.random.randint(0, 2, size=(n_samples,))

    X_test = np.random.rand(n_samples, n_dims)
    y_test = np.random.randint(0, 2, size=(n_samples,))

    clf = KNN_Classifier(k, weighted=weighted[0])
    clf._distance = lambda a, b: pairs["sk"](a, b)

    clf.fit(X, y)

    y_pred = clf.predict(X_test)

    acc1 = accuracy(y_test, y_pred)

    clf2 = KNeighborsClassifier(n_neighbors=k, algorithm="brute", weights=weighted[1], metric=pairs["sc"])
    clf2.fit(X, y)
    y_pred2 = clf2.predict(X_test)

    print(y_pred)
    print(y_pred2)

    acc2 = accuracy(y_test, y_pred2)

    assert abs(acc1 - acc2) <= epsilon


@pytest.mark.parametrize("k, weighted, pairs", itertools.product(k, weighted, distance_pairs))
def test_knn_regression(k, weighted, pairs):

    X = np.random.rand(n_samples, n_dims)
    y = np.random.rand(n_samples)

    X_test = np.random.rand(n_samples, n_dims)
    y_test = np.random.randint(0, 2, size=(n_samples,))

    clf = KNN_Regressor(k, weighted=weighted[0])
    clf._distance = lambda a, b: pairs["sk"](a, b)

    clf.fit(X, y)

    y_pred = clf.predict(X_test)

    mse1 = mse(y_test, y_pred)

    clf2 = KNeighborsRegressor(n_neighbors=k, algorithm="brute", weights=weighted[1], metric=pairs["sc"])
    clf2.fit(X, y)
    y_pred2 = clf2.predict(X_test)

    print(y_pred)
    print(y_pred2)

    mse2 = mse(y_test, y_pred2)

    assert abs(mse1 - mse2) <= epsilon
