import numpy as np
import pandas as pd
from sklearn import datasets as sk_data


def blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0,
          center_box=(-10.0, 10.0), shuffle=True, random_state=None):

    X, y = sk_data.make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std,
                              center_box=center_box, shuffle=shuffle, random_state=random_state)

    return X, y


def classification_dataset(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0,
                           scale=1.0, shuffle=True, random_state=None):

    X, y = sk_data.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                       n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes,
                                       n_clusters_per_class=n_clusters_per_class, weights=weights, flip_y=flip_y,
                                       class_sep=class_sep, hypercube=hypercube, shift=shift,
                                       scale=scale, shuffle=shuffle, random_state=random_state)

    return X, y


def iris():
    return _get_data(sk_data.load_iris())


def boston():
    return _get_data(sk_data.load_boston())


def diabetes():
    return _get_data(sk_data.load_diabetes())


def wine():
    return _get_data(sk_data.load_wine())


def tennis():
    tennis = {'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
              'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny',
                          'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
              'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild',
                              'Mild', 'Hot', 'Mild'],
              'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                           'Normal', 'High', 'Normal', 'High'],
              'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
                       'Strong', 'Strong', 'Weak', 'Strong'],
              'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
                             'No']}

    df = pd.DataFrame(tennis, dtype='category')
    dfx = df[['Day', 'Outlook', 'Temperature', 'Humidity', 'Wind']]
    dfy = df[['Day', 'PlayTennis']]
    dfx.set_index('Day', inplace=True)
    dfy.set_index('Day', inplace=True)

    return dfx, dfy


def _get_data(dataset):
    return dataset.data, dataset.target
