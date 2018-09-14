import pytest
import sys

import numpy as np

import supervised.linear_regression as skratch

from utils.optimization import StochasticGradientDescentOptimizer
from utils.preprocessing import add_dummy_feature, StandardScaler

EPSILON = 3E-2

n_samples = range(10, 400, 10)
n_features = range(1, 10)


def get_X_y_weights(n_samples, n_features, fit_intercept):

    X = np.random.rand(n_samples, n_features)
    weights = np.random.rand(X.shape[1] + fit_intercept)
    X = StandardScaler().fit_transform(X)

    features = X
    if fit_intercept:
        features = add_dummy_feature(features)

    y = features.dot(weights)

    return X, y, weights


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("fit_intercept", [0, 1])
def test_linear_regression(n_samples, n_features, fit_intercept):

    X, y, weights = get_X_y_weights(n_samples, n_features, fit_intercept)

    reg = skratch.LinearRegression(fit_intercept=fit_intercept)
    optimizer = StochasticGradientDescentOptimizer()

    reg.optimizer = optimizer
    reg.fit(X, y)
    r0 = reg.predict(X)

    error_r0 = np.mean((r0 - y)**2)

    assert error_r0 < EPSILON


@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("fit_intercept", [0, 1])
def test_analytical_linear_regression(n_samples, n_features, fit_intercept):

    X, y, weights = get_X_y_weights(n_samples, n_features, fit_intercept)

    reg = skratch.AnalyticalLinearRegression(fit_intercept=fit_intercept)

    reg.fit(X, y)
    r0 = reg.predict(X)

    error_r0 = np.mean((r0 - y)**2)

    assert error_r0 < 1E-16
