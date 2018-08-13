import numpy as np

from supervised.nb_classifier import NBClassifier


class BernoulliNB(NBClassifier):

    def _pdf(self, x, p):

        return (1.0 - x) * (1.0 - p) + x * p

    def _fit_evidence(self, X):

        feature_probas = []

        for feature in X.T:  # take the transpose to iterate through the features instead of the samples

            feature_probas.append(dict(count=len(feature[feature == 1]),
                                       n=len(feature)))

        return np.array(feature_probas)

    def _fit_likelihood(self, X, y):

        likelihood_ = []

        for c in self.classes_:

            samples = X[y == c]  # only keep samples of class c

            likelihood_.append(self._fit_evidence(samples))

        return likelihood_

    def _update_evidence(self, X):

        for i, feature in enumerate(X.T):  # take the transpose to iterate through the features instead of the samples

            self.evidence[i]["count"] += len(feature[feature == 1])
            self.evidence[i]["n"] += len(feature)

        return self.evidence_

    def _update_likelihood(self, X, y):

        for i, c in enumerate(self.classes_):

            samples = X[y == c]  # only keep samples of class c

            for i, feature in enumerate(samples.T):  # take the transpose to iterate through the features instead of the samples

                self.likelihood_[c][i]["count"] += len(feature[feature == 1])
                self.likelihood_[c][i]["n"] += len(feature)

        return self.likelihood_

    def _get_evidence(self, sample):

        evidence = 1.0

        for i, feature in enumerate(sample):

            count = self.evidence_[i]["count"]
            n = self.evidence_[i]["n"]

            evidence *= self._pdf(x=feature, p=count / n)

        return evidence

    def _get_likelihood(self, sample, c):

        likelihood = 1.0

        for i, feature in enumerate(sample):

            count = self.likelihood_[c][i]["count"]
            n = self.likelihood_[c][i]["n"]

            likelihood *= self._pdf(x=feature, p=count / n)

        return likelihood
