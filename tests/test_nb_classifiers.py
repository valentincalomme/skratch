from naive_bayes.gaussian_nb import GaussianNB
from naive_bayes.bernoulli_nb import BernoulliNB
from sklearn.naive_bayes import GaussianNB as sklearn_gnb
from sklearn.naive_bayes import BernoulliNB as sklearn_BernoulliNB

import numpy as np


def test_gaussian():
    from sklearn import datasets
    iris = datasets.load_iris()
    gnb2 = GaussianNB().fit(iris.data, iris.target)

    y_pred = gnb2.predict(iris.data)

    print(y_pred)
    print("Accuracy = {}".format(str(np.mean(y_pred == iris.target))))

    from sklearn import datasets
    iris = datasets.load_iris()
    gnb = sklearn_gnb()
    y_pred2 = gnb.fit(iris.data, iris.target).predict(iris.data)
    print(y_pred2)

    print("Accuracy = {}".format(str(np.mean(y_pred2 == iris.target))))

    assert all(y_pred == y_pred2)


def test_bernoulli():

    acc1 = []
    acc2 = []

    n_runs = 5
    n_samples = 100
    n_dims = 10

    for i in range(n_runs):

        X = np.random.randint(2, size=(n_samples, n_dims))
        y = np.random.randint(2, size=(n_samples,))

        clf = BernoulliNB()
        clf.fit(X, y)

        y_pred = clf.predict(X)

        acc1.append(np.mean(y == y_pred))

        clf2 = sklearn_BernoulliNB()
        clf2.fit(X, y)
        y_pred2 = clf2.predict(X)

        acc2.append(np.mean(y == y_pred2))

    assert abs(np.mean(acc1) - np.mean(acc2)) < 1E-2
