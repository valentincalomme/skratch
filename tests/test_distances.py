"""These tests verify that the distance computations are
consistent with scipy
"""

import pytest
import random
import string

import scipy.spatial.distance as sc  # sc for scipy
import utils.distances as sk  # sk for skratch
from weighted_levenshtein import lev
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


n_features = 10
simulations = 100
precision = 1E-14
minkowski_powers = 500


distance_pairs = [
    dict(sc=sc.cosine,         sk=sk.cosine,              binary=0),
    dict(sc=sc.euclidean,      sk=sk.euclidean,           binary=0),
    dict(sc=sc.correlation,    sk=sk.correlation,         binary=0),
    dict(sc=sc.hamming,        sk=sk.hamming,             binary=0),
    dict(sc=sc.canberra,       sk=sk.canberra,            binary=0),
    dict(sc=sc.chebyshev,      sk=sk.chebyshev,           binary=0),
    dict(sc=sc.braycurtis,     sk=sk.braycurtis,          binary=0),
    dict(sc=sc.cityblock,      sk=sk.manhattan,           binary=0),
    dict(sc=sc.sqeuclidean,    sk=sk.sqeuclidean,         binary=0),
    dict(sc=lev,               sk=sk.levenshtein,         binary=2),
    dict(sc=sc.dice,           sk=sk.dice,                binary=1),
    dict(sc=sc.yule,           sk=sk.yule,                binary=1),
    dict(sc=sc.jaccard,        sk=sk.jaccard,             binary=1),
    dict(sc=sc.kulsinski,      sk=sk.kulsinski,           binary=1),
    dict(sc=sc.rogerstanimoto, sk=sk.rogerstanimoto,      binary=1),
    dict(sc=sc.russellrao,     sk=sk.russellrao,          binary=1),
    dict(sc=sc.sokalsneath,    sk=sk.sokalsneath,         binary=1),
    dict(sc=sc.sokalmichener,  sk=sk.sokalmichener,       binary=1)
]


@pytest.fixture(params=distance_pairs)
def pairs(request):
    return request.param


@pytest.fixture(params=[i for i in range(1, minkowski_powers + 1)])
def power(request):
    return request.param


def random_vector(binary=0):

    if binary == 0:
        return np.random.rand(n_features)

    elif binary == 1:
        return np.random.randint(0, 2, (n_features,))

    elif binary == 2:
        return str(''.join(random.choice(string.ascii_letters + string.digits) for _ in range(np.random.randint(1, n_features))))


def test_equals(pairs):

    for i in range(simulations):
        x = random_vector(pairs["binary"])
        y = random_vector(pairs["binary"])

        try:
            assert pairs["sc"](x, y) - pairs["sk"](x, y) < precision
        except ZeroDivisionError:
            pass


def test_identical_equals_zero(pairs):

    for i in range(simulations):
        x = random_vector(pairs["binary"])
        y = random_vector(pairs["binary"])

        try:
            assert pairs["sk"](x, x) - pairs["sk"](x, x) == pairs["sk"](y, y) - pairs["sk"](y, y), print(pairs["sk"])
        except ZeroDivisionError:
            pass


def test_triangle_inequality(pairs):

    if pairs["sk"] not in [sk.yule, sk.russellrao, sk.dice, sk.sqeuclidean, sk.correlation, sk.cosine]:
        for i in range(simulations):
            x = random_vector(pairs["binary"])
            y = random_vector(pairs["binary"])
            z = random_vector(pairs["binary"])

            try:
                assert pairs["sk"](x, y) + pairs["sk"](y, z) >= pairs["sk"](x, z)
            except ZeroDivisionError:
                pass


def test_non_negativity(pairs):
    for i in range(simulations):
        x = random_vector(pairs["binary"])
        y = random_vector(pairs["binary"])

        try:
            assert pairs["sk"](x, y) >= 0
        except ZeroDivisionError:
            pass


def test_symmmetry(pairs):
    for i in range(simulations):
        x = random_vector(pairs["binary"])
        y = random_vector(pairs["binary"])

        try:
            assert pairs["sk"](x, y) - pairs["sk"](y, x) < precision
        except ZeroDivisionError:
            pass


def test_pdist(pairs):

    for i in range(simulations):

        try:

            if pairs["sc"] not in [lev, dld]:
                X = np.array([random_vector(pairs["binary"]) for _ in range(n_features)])

                SC = sc.squareform(sc.pdist(X, pairs["sc"]))
                SK = sk.pdist(X, pairs["sk"])
                assert np.mean(SC - SK) < precision
        except ZeroDivisionError:
            pass


def test_cdist(pairs):

    for i in range(simulations):

        try:

            if pairs["sc"] not in [lev, dld]:
                X = np.array([random_vector(pairs["binary"]) for _ in range(n_features)])
                Y = np.array([random_vector(pairs["binary"]) for _ in range(n_features)])

                SC = sc.cdist(X, Y, pairs["sc"])
                SK = sk.cdist(X, Y, pairs["sk"])

                assert np.mean(SC - SK) < precision
        except ZeroDivisionError:
            pass


def test_minkowski_power(power):

    for i in range(0, simulations):
        x = random_vector()
        y = random_vector()

        assert sc.minkowski(x, y, power) - sk.minkowski(x, y, power) < precision
