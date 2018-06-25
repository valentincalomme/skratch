import numpy as np
from sklearn import datasets


def boston():
    return datasets.load_boston(return_X_y=True)


def diabetes():
    return datasets.load_diabetes(return_X_y=True)


def regression(n_samples=100,
               n_features=100,
               n_informative=10,
               n_targets=1,
               bias=0.0,
               effective_rank=None,
               tail_strength=0.5,
               noise=0.0,
               shuffle=True,
               coef=False,
               random_state=None):

    return datasets.make_regression(n_samples=n_samples,
                                    n_features=n_features,
                                    n_informative=n_informative,
                                    n_targets=n_targets,
                                    bias=bias,
                                    effective_rank=effective_rank,
                                    tail_strength=tail_strength,
                                    noise=noise,
                                    shuffle=shuffle,
                                    coef=coef,
                                    random_state=random_state)


def friedman1(n_samples=100,
              n_features=10,
              noise=0.0,
              random_state=None):

    return datasets.make_friedman1(n_samples=n_samples,
                                   n_features=n_features,
                                   noise=noise,
                                   random_state=random_state)


def friedman2(n_samples=100,
              noise=0.0,
              random_state=None):

    return datasets.make_friedman2(n_samples=n_samples,
                                   noise=noise,
                                   random_state=random_state)


def friedman3(n_samples=100,
              noise=0.0,
              random_state=None):

    return datasets.make_friedman3(n_samples=n_samples,
                                   noise=noise,
                                   random_state=random_state)
