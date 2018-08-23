import itertools

import pytest
import numpy as np

from sklearn.feature_extraction import DictVectorizer
import sklearn.preprocessing as sklearn
import utils.preprocessing as skratch


EPSILON = 1E-16
n_samples = range(10, 100, 10)
n_features = range(10, 100, 10)
threshold = np.linspace(0, 1, 5)
norm = ["l2", "l1", "max"]
degree = range(2, 5)


@pytest.mark.parametrize("n", range(3, 20))
def test_label_binarizer_fit_transform(n):

    max_value = 10
    y = np.array([np.random.randint(max_value) for _ in range(n)])

    enc0 = skratch.LabelBinarizer()
    ENC0 = enc0.fit_transform(y)
    enc1 = sklearn.LabelBinarizer()
    ENC1 = enc1.fit_transform(y)

    assert (ENC0 == ENC1).all()


@pytest.mark.parametrize("n", range(3, 20))
def test_label_binarizer_inverse_transform(n):

    max_value = 10
    y = np.array([np.random.randint(max_value) for _ in range(n)])

    enc0 = skratch.LabelBinarizer()
    ENC0 = enc0.fit_transform(y)

    print(y)
    print(enc0.inverse_transform(ENC0))

    assert (enc0.inverse_transform(ENC0) == y).all()


@pytest.mark.parametrize("n", range(3, 20))
def test_label_encoder_fit_transform(n):

    max_value = 10
    y = np.array([np.random.randint(max_value) for _ in range(n)])

    enc0 = skratch.LabelEncoder()
    ENC0 = enc0.fit_transform(y)
    enc1 = sklearn.LabelEncoder()
    ENC1 = enc1.fit_transform(y)

    assert (ENC0 == np.array(ENC1)).all()


@pytest.mark.parametrize("n", range(3, 20))
def test_label_encoder_inverse_transform(n):

    max_value = 10
    y = np.array([np.random.randint(max_value) for _ in range(n)])

    enc0 = skratch.LabelEncoder()
    ENC0 = enc0.fit_transform(y)

    print(y)
    print(enc0.inverse_transform(ENC0))

    assert (enc0.inverse_transform(ENC0) == y).all()


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
def test_add_dummy_feature(n_samples, n_features):

    X = np.random.rand(n_samples, n_features)

    new_skratch = skratch.add_dummy_feature(X)
    new_sklearn = sklearn.add_dummy_feature(X)

    assert (new_sklearn == new_skratch).all()


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("threshold", threshold)
def test_binarize(n_samples, n_features, threshold):

    X = np.random.rand(n_samples, n_features)

    new_skratch = skratch.binarize(X)
    new_sklearn = sklearn.binarize(X)

    assert (new_sklearn == new_skratch).all()


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
def test_MaxAbsScaler(n_samples, n_features):

    X = np.random.rand(n_samples, n_features)

    new_skratch = skratch.MaxAbsScaler().fit_transform(X)
    new_sklearn = sklearn.MaxAbsScaler().fit_transform(X)

    assert (new_sklearn == new_skratch).all()


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
def test_MinMaxScaler(n_samples, n_features):

    X = np.random.rand(n_samples, n_features)

    new_skratch = skratch.MinMaxScaler().fit_transform(X)
    new_sklearn = sklearn.MinMaxScaler().fit_transform(X)

    assert np.mean(new_sklearn - new_skratch) < EPSILON


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("norm", norm)
def test_Normalizer(n_samples, n_features, norm):

    X = np.random.rand(n_samples, n_features)

    new_skratch = skratch.Normalizer(norm).fit_transform(X)
    new_sklearn = sklearn.Normalizer(norm).fit_transform(X)

    assert np.mean(new_sklearn - new_skratch) < EPSILON


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("with_mean", [0, 1])
@pytest.mark.parametrize("with_std", [0, 1])
def test_StandardScaler(n_samples, n_features, with_mean, with_std):

    X = np.random.rand(n_samples, n_features)

    new_skratch = skratch.StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(X)
    new_sklearn = sklearn.StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(X)

    assert np.mean(new_sklearn - new_skratch) < EPSILON


@pytest.mark.parametrize("n_samples", range(10, 100, 10))
@pytest.mark.parametrize("n_features", range(2, 10))
@pytest.mark.parametrize("degree", degree)
def test_PolynomialFeatures(n_samples, n_features, degree):

    X = np.random.rand(n_samples, n_features)

    new_skratch = skratch.polynomial_features(X, degree)
    new_sklearn = sklearn.PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)

    assert (new_sklearn == new_skratch).all()
