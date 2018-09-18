import numpy as np

from supervised.nb_classifier import NBClassifier

EPSILON = 1E-16  # offset to avoid "divide by zero" errors


class GaussianNB(NBClassifier):

    def _pdf(self, x, mean, std):

        num = np.exp(-((x - mean)**2) / (EPSILON + 2 * std**2))
        den = np.sqrt(2 * np.pi * std**2) + EPSILON

        return num / den

    def _fit_evidence(self, X):

        feature_probas = []

        for feature in X.T:  # iterate through the features instead of the samples

            feature_probas.append(dict(mean=np.mean(feature),
                                       n=len(feature),
                                       std=np.std(feature, ddof=1)))

        return np.array(feature_probas)

    def _fit_likelihood(self, X, y):

        likelihood_ = []

        for c in self.classes_:

            samples = X[y == c]  # only keep samples of class c

            likelihood_.append(self._fit_evidence(samples))

        return np.array(likelihood_)

    def _get_evidence(self, sample):

        evidence = 1.0

        for i, feature in enumerate(sample):

            mean = self.evidence_[i]["mean"]
            std = self.evidence_[i]["std"]

            evidence *= self._pdf(feature, mean, std)

        return evidence

    def _get_likelihood(self, sample, c):

        likelihood = 1.0

        for i, feature in enumerate(sample):

            mean = self.likelihood_[c][i]["mean"]
            std = self.likelihood_[c][i]["std"]

            likelihood *= self._pdf(feature, mean, std)

        return likelihood

    def _update_evidence(self, X):

        for i, feature in enumerate(X.T):   # iterate through the features instead of the samples

            self.evidence_[i] = self._update_mean_std_n(feature, self.evidence_[i])

        return self.evidence_

    def _update_likelihood(self, X, y):

        for c in self.classes_:

            samples = X[y == c]  # only keep samples of class c

            for i, feature in enumerate(samples.T):  # iterate through the features instead of the samples

                self.likelihood_[c][i] = self._update_mean_std_n(feature, self.likelihood_[c][i])

        return self.likelihood_

    def _update_mean_std_n(self, feature, mean_std_n):

        old_m = mean_std_n["mean"]
        old_std = mean_std_n["std"]
        old_n = mean_std_n["n"]

        n = old_n + len(feature)

        m = (old_m * old_n + np.mean(feature) * n) / (old_n + n)

        s = np.sqrt((old_n * (old_std**2 + (old_m - m)**2)
                     + len(feature) * (np.var(feature)
                                       + (np.mean(feature) - m)**2)
                     ) / (old_n + len(feature)))

        return dict(mean=m, std=std, n=n)
