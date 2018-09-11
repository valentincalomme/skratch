import pytest
import sys

import numpy as np

import supervised.linear_regression as skratch
import sklearn.linear_model as sklearn

from utils.optimization import GradientDescentOptimizer
from utils.regularization import LASSO, Ridge, ElasticNet
from utils.preprocessing import add_dummy_feature, StandardScaler

EPSILON = 1E-4

n_samples = range(10, 100, 10)
n_features = range(1, 10)
_lambda = np.linspace(0.1, 0.5, num=5)
ratio = np.linspace(0.1, 0.9, num=5)
tol = 1E-8
learning_rate = 0.1
momentum = 0


def get_X_y(n_samples, n_features, fit_intercept):

    X = np.random.rand(n_samples, n_features)
    weights = np.random.rand(X.shape[1] + fit_intercept)
    X = StandardScaler().fit_transform(X)

    features = X
    if fit_intercept:
        features = add_dummy_feature(features)

    y = features.dot(weights)

    return X, y

# @pytest.mark.parametrize("n_samples", n_samples)
# @pytest.mark.parametrize("n_features", n_features)
# @pytest.mark.parametrize("fit_intercept", [0, 1])
# @pytest.mark.parametrize("_lambda", _lambda)
# def test_lasso_linear_regression(n_samples, n_features, fit_intercept, _lambda):

#     X, y = get_X_y(n_samples, n_features, fit_intercept)

#     optimizer = GradientDescentOptimizer(learning_rate, momentum)
#     regularizer = LASSO(_lambda)

#     r0 = skratch.LinearRegression(optimizer, regularizer=regularizer, tol=tol,
#                                   fit_intercept=fit_intercept).fit(X, y).predict(X)
#     r1 = sklearn.Lasso(alpha=_lambda, tol=tol, fit_intercept=fit_intercept).fit(X, y).predict(X)

#     error_r0 = np.mean((r0 - y)**2)
#     error_r1 = np.mean((r1 - y)**2)

#     assert np.abs(error_r0 - error_r1) < EPSILON

    # assert np.mean(np.abs(r0 - r1) / (r1 + 1E-18)) < EPSILON

# @pytest.mark.parametrize("n_samples", n_samples)
# @pytest.mark.parametrize("n_features", n_features)
# @pytest.mark.parametrize("fit_intercept", [0])
# @pytest.mark.parametrize("_lambda", _lambda)
# def test_ridge_linear_regression(n_samples, n_features, fit_intercept, _lambda):

#     X, y = get_X_y(n_samples, n_features, fit_intercept)

#     optimizer = GradientDescentOptimizer(learning_rate=0.01, momentum=0.1)
#     regularizer = Ridge(_lambda)

#     r0 = skratch.LinearRegression(optimizer, regularizer=regularizer, fit_intercept=fit_intercept).fit(X, y).predict(X)
#     r1 = sklearn.Ridge(alpha=_lambda,  tol=1E-8, fit_intercept=fit_intercept).fit(X, y).predict(X)

#     assert np.mean((r0 - r1) / (r1+1E-18)) < EPSILON


# @pytest.mark.parametrize("n_samples", n_samples)
# @pytest.mark.parametrize("n_features", n_features)
# @pytest.mark.parametrize("fit_intercept", [0])
# @pytest.mark.parametrize("_lambda", _lambda)
# @pytest.mark.parametrize("ratio", ratio)
# def test_elastic_net_linear_regression(n_samples, n_features, fit_intercept, _lambda, ratio):

#     X, y = get_X_y(n_samples, n_features, fit_intercept)

#     optimizer = GradientDescentOptimizer(learning_rate=0.01, momentum=0.1)
#     regularizer = ElasticNet(_lambda, ratio)

#     r0 = skratch.LinearRegression(optimizer, regularizer=regularizer, fit_intercept=fit_intercept).fit(X, y).predict(X)
#     r1 = sklearn.ElasticNet(alpha=_lambda,  tol=1E-8, l1_ratio=ratio, fit_intercept=fit_intercept).fit(X, y).predict(X)

#     assert np.mean((r0 - r1) / (r1+1E-18)) < EPSILON
