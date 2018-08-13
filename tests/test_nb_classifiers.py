import pytest

import scipy.stats as ss
from sklearn import datasets
import numpy as np


EPSILON = 5E-2
N_SAMPLES = [500]
N_DIMS = range(2, 10)
N_CLASSES = range(2, 5)
N = 100


######################################################################
######################################################################
######################################################################


from supervised.gaussian_nb import GaussianNB
from sklearn.naive_bayes import GaussianNB as sklearn_gnb


@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_dims", N_DIMS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_gaussian_vs_sklearn(n_samples, n_dims, n_classes):

    X = np.random.rand(n_samples, n_dims)
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = GaussianNB()
    clf2 = sklearn_gnb()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert np.mean(y_pred1 == y_pred2) >= 1 - EPSILON


@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_dims", N_DIMS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_gaussian_is_not_stochastic(n_samples, n_dims, n_classes):

    X = np.random.rand(n_samples, n_dims)
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = GaussianNB()
    clf2 = GaussianNB()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert all(y_pred1 == y_pred2)


@pytest.mark.parametrize("x", np.random.rand(int(np.round(np.power(float(N), 1 / 3)))))
@pytest.mark.parametrize("loc", np.random.rand(int(np.round(np.power(float(N), 1 / 3)))))
@pytest.mark.parametrize("scale", np.random.rand(int(np.round(np.power(float(N), 1 / 3)))))
def test_gaussian_pdf(x, loc, scale):

    skratch_pdf = GaussianNB()._pdf
    scipy_pdf = ss.norm.pdf

    assert skratch_pdf(x, loc, scale) - scipy_pdf(x, loc, scale) <= 1E-12

######################################################################
######################################################################
######################################################################


from supervised.bernoulli_nb import BernoulliNB
from sklearn.naive_bayes import BernoulliNB as sklearn_BernoulliNB


@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_dims", N_DIMS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_bernoulli_vs_sklearn(n_samples, n_dims, n_classes):

    X = np.random.randint(2, size=(n_samples, n_dims))
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = BernoulliNB()
    clf2 = sklearn_BernoulliNB()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert np.mean(y_pred1 == y_pred2) >= 1 - EPSILON


@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_dims", N_DIMS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_bernoulli_is_not_stochastic(n_samples, n_dims, n_classes):

    X = np.random.randint(2, size=(n_samples, n_dims))
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = BernoulliNB()
    clf2 = BernoulliNB()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert all(y_pred1 == y_pred2)


@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("p", np.random.rand(N // 2))
def test_bernoulli_pdf(x, p):

    skratch_pdf = BernoulliNB()._pdf
    scipy_pdf = ss.bernoulli.pmf

    assert skratch_pdf(x, p) - scipy_pdf(x, p) <= 1E-12


######################################################################
######################################################################
######################################################################

from supervised.multinomial_nb import MultinomialNB
from sklearn.naive_bayes import MultinomialNB as sklearn_MultinomialNB

n = 10


@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_dims", N_DIMS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("alpha", np.random.rand(10))
def test_multinomial_vs_sklearn(n_samples, n_dims, n_classes, alpha):

    X = np.random.randint(10, size=(n_samples, n_dims))
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = MultinomialNB(alpha=alpha)
    clf2 = sklearn_MultinomialNB(alpha=alpha)

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert np.mean(y_pred1 == y_pred2) >= 1 - EPSILON


@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_dims", N_DIMS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_multinomial_is_not_stochastic(n_samples, n_dims, n_classes):

    X = np.random.randint(10, size=(n_samples, n_dims))
    y = np.random.randint(0, n_classes, size=(n_samples,))

    clf1 = MultinomialNB()
    clf2 = MultinomialNB()

    y_pred1 = clf1.fit(X, y).predict(X)
    y_pred2 = clf2.fit(X, y).predict(X)

    assert all(y_pred1 == y_pred2)


def _gen_multinomial_pdf_parameters():
    for i in range(N):

        n_integers = np.random.randint(2, 10)

        x = np.random.randint(1, 10, n_integers)
        p = np.random.dirichlet(np.ones(n_integers), size=1)[0]

        yield x, p


@pytest.mark.parametrize("x, p", _gen_multinomial_pdf_parameters())
def test_multinomial_pdf(x, p):

    skratch_pdf = MultinomialNB()._pdf
    scipy_pdf = lambda x, p: ss.multinomial.pmf(x, np.sum(x), p)

    assert skratch_pdf(x, p) - scipy_pdf(x, p) <= 1E-5
