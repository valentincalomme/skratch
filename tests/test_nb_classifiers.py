import pytest
from itertools import product

from sklearn import datasets
import numpy as np

from supervised.naive_bayes.gaussian_nb import GaussianNB
from sklearn.naive_bayes import GaussianNB as sklearn_gnb

from supervised.naive_bayes.bernoulli_nb import BernoulliNB
from sklearn.naive_bayes import BernoulliNB as sklearn_BernoulliNB

from supervised.naive_bayes.multinomial_nb import MultinomialNB
from sklearn.naive_bayes import MultinomialNB as sklearn_MultinomialNB

EPSILON = 5E-2
N_SAMPLES = [500]
N_DIMS = range(2, 10)
N_CLASSES = range(2, 5)


@pytest.mark.parametrize("n_samples, n_dims, n_classes", product(N_SAMPLES, N_DIMS, N_CLASSES))
def test_gaussian_vs_sklearn(n_samples, n_dims, n_classes):

    X = np.random.rand(n_samples, n_dims)
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = GaussianNB()
    clf2 = sklearn_gnb()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert np.mean(y_pred1 == y_pred2) >= 1 - EPSILON


@pytest.mark.parametrize("n_samples, n_dims, n_classes", product(N_SAMPLES, N_DIMS, N_CLASSES))
def test_bernoulli_vs_sklearn(n_samples, n_dims, n_classes):

    X = np.random.randint(2, size=(n_samples, n_dims))
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = BernoulliNB()
    clf2 = sklearn_BernoulliNB()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert np.mean(y_pred1 == y_pred2) >= 1 - EPSILON




@pytest.mark.parametrize("n_samples, n_dims, n_classes", product(N_SAMPLES, N_DIMS, N_CLASSES))
def test_gaussian_is_not_stochastic(n_samples, n_dims, n_classes):

    X = np.random.rand(n_samples, n_dims)
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = GaussianNB()
    clf2 = GaussianNB()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert all(y_pred1 == y_pred2)


@pytest.mark.parametrize("n_samples, n_dims, n_classes", product(N_SAMPLES, N_DIMS, N_CLASSES))
def test_bernoulli_is_not_stochastic(n_samples, n_dims, n_classes):

    X = np.random.randint(2, size=(n_samples, n_dims))
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = BernoulliNB()
    clf2 = BernoulliNB()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert all(y_pred1 == y_pred2)


# @pytest.mark.parametrize("n_samples, n_dims, n_classes", product(N_SAMPLES, N_DIMS, N_CLASSES))
# def test_multinomial_vs_sklearn(n_samples, n_dims, n_classes):

#     X = np.random.randint(2, size=(n_samples, n_dims))
#     y = np.random.randint(0, n_classes, size=(n_samples,))

#     clf1 = MultinomialNB()
#     clf2 = sklearn_MultinomialNB()

#     y_pred1 = clf1.fit(X, y).predict(X)
#     y_pred2 = clf2.fit(X, y).predict(X)

#     assert np.mean(y_pred1 == y_pred2) >= 1 - EPSILON

# @pytest.mark.parametrize("n", range(3,10))
# def test_multinomial_distribution(n):

#     from supervised.naive_bayes.multinomial_nb import MultinomialNB
#     from sklearn.naive_bayes import MultinomialNB as sklearn_MultinomialNB
#     import numpy as np
#     import scipy.stats as ss
#     from math import factorial as fact

#     def pdf(x, p):

#         n = np.sum(x)

#         num = fact(n) * np.product(list(map(lambda _: _[0]**_[1], zip(p, x))))
#         den = np.product(list(map(fact, x)))

#         return num / den

#     def _pdf(x, p):

#         return ss.multinomial(np.sum(x), p).pmf(x)

#     x = np.array([np.random.randint(1,10) for _ in range(3)])
#     print(x)
#     p = [0.1, 0.2, 0.7]
#     assert abs(pdf(x, p) - _pdf(x, p)) < 1E-3
