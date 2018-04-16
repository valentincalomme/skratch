import numpy as np

from supervised.naive_bayes.nb_classifier import NBClassifier


class BernoulliNB(NBClassifier):

    def _pdf(self, x, p):

        return (1.0 - x) * (1.0 - p) + x * p

    def _fit_likelihood(self, X, y):

        likelihood_ = {}

        for c in self.classes_:

            samples = X[y == c]

            likelihood_[c] = [dict(count=np.sum(feature),
                                   n=len(feature))
                              for feature in samples.transpose()]

        return likelihood_

    def _update_likelihood(self, X, y):

        likelihoods = {}

        for i, c in enumerate(self.classes_):
            samples = X[y == c]

            for i, feature in enumerate(samples.transpose()):

                count = self.evidence[i]["count"] + np.sum(feature)
                n = self.evidence[i]["n"] + len(feature)

                self.likelihood_[c][i] = dict(count=count, n=n)

        return self.likelihood_

    def _get_likelihood(self, sample, c):

        likelihood = 1.0

        for i, feature in enumerate(sample):

            count = self.likelihood_[c][i]["count"]
            n = self.likelihood_[c][i]["n"]

            likelihood *= self._pdf(x=feature, p=count / n)

        return likelihood

    def _fit_evidence(self, X):

        feature_probas = [(dict(count=np.sum(feature),
                                n=len(feature)))
                          for feature in X.transpose()]

        return np.array(feature_probas)

    def _get_evidence(self, sample):

        evidence = 1.0

        for i, feature in enumerate(sample):

            count = self.evidence_[i]["count"]
            n = self.evidence_[i]["n"]

            evidence += self._pdf(x=feature, p=count / n)

        return evidence

    def _update_evidence(self, X):

        for i, feature in enumerate(X.transpose()):

            count = self.evidence[i]["count"] + np.sum(feature)
            n = self.evidence[i]["n"] + len(feature)

            self.evidence_[i] = dict(count=count, n=n)

        return self.evidence_
