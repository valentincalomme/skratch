import numpy as np

from supervised.naive_bayes.nb_classifier import NBClassifier


class GaussianNB(NBClassifier):

    def _pdf(self, x, mean, std):

        num = np.exp(-((x - mean)**2) / (2 * std**2))
        den = np.sqrt(2 * np.pi * std**2)

        return num / den

    def _fit_likelihood(self, X, y):

        likelihood_ = {}

        for c in self.classes_:
            samples = X[y == c]

            likelihood_[c] = [dict(mean=np.mean(feature),
                                   std=np.std(feature, ddof=1),
                                   n=len(feature)) for feature in samples.T]

        return likelihood_

    def _get_likelihood(self, sample, c):

        likelihood = 1.0

        for i, feature in enumerate(sample):

            mean = self.likelihood_[c][i]["mean"]
            std = self.likelihood_[c][i]["std"]

            likelihood *= self._pdf(feature, mean, std)

        return likelihood

    def _update_likelihood(self, X, y):

        likelihoods = {}

        for c in enumerate(self.classes_):

            feature_probas = []
            samples = X[y == c]

            for i, x in enumerate(samples.T):

                old_m = self.likelihood_[c][i]["mean"]
                old_std = self.likelihood_[c][i]["std"]
                old_n = self.likelihood_[c][i]["n"]

                n = old_n + len(x)

                m = (old_m * old_n + np.mean(x) * n) / (old_n + n)

                s = np.sqrt((old_n * (old_std**2 + (old_m - m)**2) + len(x) *
                             (np.var(x) + (np.mean(x) - m)**2)) / (old_n + len(x)))

                self.likelihood_[c][i] = dict(mean=m, std=s, n=n)

        return self.likelihood_

    def _fit_evidence(self, X):

        feature_probas = []

        for feature in X.T:

            m = np.mean(feature)
            n = len(feature)
            s = np.std(feature, ddof=1)

            feature_probas.append(dict(mean=m, std=s, n=n))

        return np.array(feature_probas)

    def _update_evidence(self, X):

        for i, feature in enumerate(X.T):

            old_m = self.evidence_[i]["mean"]
            old_std = self.evidence_[i]["std"]
            old_n = self.evidence_[i]["n"]

            n = old_n + len(feature)

            m = (old_m * old_n + np.mean(feature) * n) / (old_n + n)

            s = np.sqrt((old_n * (old_std**2 + (old_m - m)**2)
                         + len(feature) * (np.var(feature)
                                           + (np.mean(feature) - m)**2)
                         ) / (old_n + len(feature)))

            self.evidence_[i] = dict(mean=m, std=s, n=n)

        return self.evidence_

    def _get_evidence(self, sample):

        evidence = 1.0

        for i, feature in enumerate(sample):

            mean = self.evidence_[i]["mean"]
            std = self.evidence_[i]["std"]

            evidence *= self._pdf(feature, mean, std)

        return evidence
