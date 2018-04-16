from math import factorial as fact
from collections import Counter

import numpy as np

from supervised.naive_bayes.nb_classifier import NBClassifier


class MultinomialNB(NBClassifier):

    def __init__(self, smoothing=1.0):

        super().__init__()
        self.smoothing = smoothing

    def _pdf(self, x, p):

        assert len(x) == len(p)

        n = np.sum(x)

        num = fact(n) * np.product(list(map(lambda _: _[0]**_[1], zip(p, x))))
        den = np.product(list(map(fact, x)))

        return num / den

    def _fit_likelihood(self, X, y):

        likelihood_ = {}

        for c in self.classes_:

            samples = X[y == c]

            likelihood_[c] = [dict(counts=dict(Counter(feature)))
                              for feature in samples.T]

        return likelihood_

    def _update_likelihood(self, X, y):

        for c in enumerate(self.classes_):
            samples = X[y == c]

            for i, feature in enumerate(samples.T):

                old_counts = self.likelihood_[c][i]["counts"]
                new_counts = {v: c for v, c in np.unique(feature, return_counts=True)}

                self.likelihood_[c][i] = dict(counts=dict(Counter(old_counts) + Counter(new_counts)))

        return self.likelihood_

    def _get_likelihood(self, sample, c):

        p = []

        for i, feature in enumerate(sample):

            counts = self.likelihood_[c][i]["counts"]

            x = 0
            if feature in counts:
                x = counts[feature]
            N = sum(counts.values())
            d = len(sample)
            a = self.smoothing

            p.append((x + a) / (N + (a * d)))


        # print("p:", p)
        # print("sample:", sample)
        return self._pdf(sample, p)

    def _fit_evidence(self, sample):

        evidence_ = [dict(counts=dict(Counter(feature)))
                     for feature in sample.T]

        return evidence_

    def _update_evidence(self, X):

        for i, feature in enumerate(X.T):

            old_counts = self.evidence_[c][i]["counts"]
            new_counts = {v: c for v, c in np.unique(feature, return_counts=True)}

            self.evidence_[i] = dict(counts=dict(Counter(old_counts) + Counter(new_counts)))

        return self.evidence_

    def _get_evidence(self, sample):

        p = []

        for i, feature in enumerate(sample):

            counts = self.evidence_[i]["counts"]

            x = 0
            if feature in counts:
                x = counts[feature]
            N = sum(counts.values())
            d = len(sample)
            a = self.smoothing

            p.append((x + a) / (N + a * d))

        return self._pdf(sample, p)
