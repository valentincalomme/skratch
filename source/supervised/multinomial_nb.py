from math import factorial as fact
from collections import Counter

import scipy.stats as ss
import numpy as np

from supervised.nb_classifier import NBClassifier


class MultinomialNB(NBClassifier):

    def __init__(self, alpha=1.0):

        super().__init__()
        self.alpha = alpha

    def _pdf(self, x, p):

        f = fact(np.sum(x))

        for P, X in zip(p, x):
            f *= (P**X) / fact(X)

        return f

    def _fit_evidence(self, X):

        evidence_ = np.sum(X, axis=0)

        return evidence_

    def _fit_likelihood(self, X, y):

        likelihood_ = []

        for c in self.classes_:
            samples = X[y == c]   # only keep samples of class c

            likelihood_.append(self._fit_evidence(samples))

        return likelihood_

    def _get_evidence(self, sample):

        p = []

        for i, feature in enumerate(sample):

            x = self.evidence_[i]
            N = np.sum(self.evidence_)
            d = len(sample)
            a = self.alpha

            prob = (x + a) / (N + (a * d))

            p.append(prob)

        return self._pdf(sample, p)

    def _get_likelihood(self, sample, c):

        p = []

        for i, feature in enumerate(sample):

            x = self.likelihood_[c][i]
            N = np.sum(self.likelihood_[c])
            d = len(sample)
            a = self.alpha

            prob = (x + a) / (N + (a * d))

            p.append(prob)

        return self._pdf(sample, p)

    def _update_evidence(self, X):

        self.evidence_ += np.sum(X, axis=0)

        return self.evidence_

    def _update_likelihood(self, X, y):

        for i, c in enumerate(self.classes_):
            samples = X[y == c]   # only keep samples of class c

            self.likelihood_[i] += np.sum(samples, axis=0)

        return likelihood_
