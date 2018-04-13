def test_gaussian():

    from naive_bayes.gaussian_nb import GaussianNB
    from sklearn.naive_bayes import GaussianNB as sklearn_gnb
    from sklearn import datasets
    import numpy as np

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
    from naive_bayes.bernoulli_nb import BernoulliNB
    from sklearn.naive_bayes import BernoulliNB as sklearn_BernoulliNB
    import numpy as np

    acc1 = []
    acc2 = []

    n_runs = 5
    n_samples = 100
    n_dims = 10

    for i in range(n_runs):

        X = np.random.randint(2, size=(n_samples, n_dims))
        y = np.random.randint(1, 5, size=(n_samples,))

        clf = BernoulliNB()
        clf.fit(X, y)

        y_pred = clf.predict(X)

        acc1.append(np.mean(y == y_pred))

        clf2 = sklearn_BernoulliNB()
        clf2.fit(X, y)
        y_pred2 = clf2.predict(X)

        acc2.append(np.mean(y == y_pred2))

    assert abs(np.mean(acc1) - np.mean(acc2)) < 1E-2


def test_multinomial():
    from naive_bayes.multinomial_nb import MultinomialNB
    from sklearn.naive_bayes import MultinomialNB as sklearn_MultinomialNB
    import numpy as np

    acc1 = []
    acc2 = []

    n_runs = 1
    n_samples = 10
    n_dims = 5

    for i in range(n_runs):

        X = np.random.randint(3, size=(n_samples, n_dims))
        y = np.random.randint(0, 2, size=(n_samples,))

        clf = MultinomialNB()
        clf.fit(X, y)


        probas  = clf.predict_proba(X)

        for i, x in enumerate(X):
            print("X:", X)
            print("Probas:", probas)
        y_pred = clf.predict(X)


        acc1.append(np.mean(y == y_pred))

        clf2 = sklearn_MultinomialNB()
        clf2.fit(X, y)
        y_pred2 = clf2.predict(X)

        print() 
        print(clf2.predict_proba(X))

        acc2.append(np.mean(y == y_pred2))

    assert abs(np.mean(acc1) - np.mean(acc2)) < 1E-1

def test_multinomial_distribution():
    from naive_bayes.multinomial_nb import MultinomialNB
    from sklearn.naive_bayes import MultinomialNB as sklearn_MultinomialNB
    import numpy as np
    import scipy.stats as ss
    from math import factorial as fact

    def pdf(x, p):

        n = np.sum(x)

        num = fact(n) * np.product(list(map(lambda _: _[0]**_[1], zip(p, x))))
        den = np.product(list(map(fact, x)))

        return num / den

    def _pdf(x, p):

        return ss.multinomial(np.sum(x), p).pmf(x)


    x = [2,2,2]
    p = [0.1,0.2,0.7]
    assert abs(pdf(x,p) - _pdf(x,p)) < 1E-3