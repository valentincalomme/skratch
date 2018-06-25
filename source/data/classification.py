import numpy as np
from sklearn import datasets


def iris():
    return datasets.load_iris(return_X_y=True)


def wine():
    return datasets.load_wine(return_X_y=True)


def digits(n_class=10):
    return datasets.load_digits(n_class=10,
                                return_X_y=True)


def breast_cancer():
    return datasets.load_breast_cancer(return_X_y=True)


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
                               random_state=random_state)


def classification(n_samples=100,
                   n_features=20,
                   n_informative=2,
                   n_redundant=2,
                   n_repeated=0,
                   n_classes=2,
                   n_clusters_per_class=2,
                   weights=None,
                   flip_y=0.01,
                   class_sep=1.0,
                   hypercube=True,
                   shift=0.0,
                   scale=1.0,
                   shuffle=True,
                   random_state=None):

    return datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features,
                                        n_informative=n_informative,
                                        n_redundant=n_redundant,
                                        n_repeated=n_repeated,
                                        n_classes=n_classes,
                                        n_clusters_per_class=n_clusters_per_class,
                                        weights=weights,
                                        flip_y=flip_y,
                                        class_sep=class_sep,
                                        hypercube=hypercube,
                                        shift=shift,
                                        scale=scale,
                                        shuffle=shuffle,
                                        random_state=random_state)


def gaussian_quantiles(mean=None,
                       cov=1.0,
                       n_samples=100,
                       n_features=2,
                       n_classes=3,
                       shuffle=True,
                       random_state=None):

    return datasets.make_gaussian_quantiles(mean=mean,
                                            cov=cov,
                                            n_samples=n_samples,
                                            n_features=n_features,
                                            n_classes=n_classes,
                                            shuffle=shuffle,
                                            random_state=random_state)


def hastie(n_samples=12000,
           random_state=None):
    return datasets.make_hastie_10_2(n_samples=n_samples,
                                     random_state=random_state)


def circles(n_samples=100,
            shuffle=True,
            noise=None,
            random_state=None,
            factor=0.8):

    return datasets.make_circles(n_samples=n_samples,
                                 shuffle=shuffle,
                                 noise=noise,
                                 random_state=random_state,
                                 factor=factor)


def moons(n_samples=100,
          shuffle=True,
          noise=None,
          random_state=None,
          ):

    return datasets.make_moons(n_samples=n_samples,
                               shuffle=shuffle,
                               noise=noise,
                               random_state=random_state)
