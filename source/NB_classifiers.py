import numpy as np
import scipy.stats as ss


class NBClassifier(object):

    def predict(self, X, y=None):

        joint_probas = self._predict_joint_proba(X)
        indices = np.argmax(joint_probas, axis=1)

        return self.classes_[indices]

    def _predict_joint_proba(self, X, y=None):

        return np.array([[self._get_prior(c) * self._get_likelihood(sample, c) for c in self.classes_]
                         for sample in X])

    def predict_proba(self, X, y=None):

        joint_probas = self._predict_joint_proba(X, y)
        evidence = np.array([[self._get_evidence(x)] for x in X])

        return joint_probas / evidence

    def fit(self, X, y):

        self.priors_ = self._fit_prior(y)
        self.evidence_ = self._fit_evidence(X)
        self.likelihood_ = self._fit_likelihood(X, y)

        return self

    def update(self, X, y):

        self.priors_ = self._update_priors(y)
        self.evidence_ = self._update_evidence(X)
        self.likelihood_ = self._update_likelihood(X, y)

        return self

    def _fit_prior(self, y):

        self.classes_, counts = np.unique(y, return_counts=True)
        total_count = np.sum(counts)

        return {c: dict(count=counts[i], total_count=total_count) for i, c in enumerate(self.classes_)}

    def _get_prior(self, c):

        count = self.priors_[c]["count"]
        total_count = self.priors_[c]["total_count"]

        return count / total_count

    def _update_priors(self, y):

        self.classes_, counts = np.unique(y, return_counts=True)
        total_count = np.sum(counts)

        for i, c in enumerate(self.classes_):

            self.priors_[c]["count"] += counts[i]
            self.priors_[c]["total_count"] += total_count

        return self.priors_


class GaussianNB(NBClassifier):

    def _pdf(self, x, mean, std):

        num = np.exp(-((x - mean)**2) / (2 * std**2))
        den = np.sqrt(2 * np.pi * std**2)

        return num / den

    def _fit_evidence(self, X):

        feature_probas = []

        for feature in X.transpose():

            m = np.mean(feature)
            n = len(feature)
            s = np.std(feature, ddof=1)

            feature_probas.append(dict(mean=m, std=s, n=n))

        return np.array(feature_probas)

    def _update_evidence(self, X):

        for i, feature in enumerate(X.transpose()):

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

        evidence = 1

        for i, feature in enumerate(sample):

            mean = self.evidence_[i]["mean"]
            std = self.evidence_[i]["std"]

            evidence *= self._pdf(feature, mean, std)

        return evidence

    def _fit_likelihood(self, X, y):

        self.likelihood_ = []

        for i, c in enumerate(self.classes_):
            samples = X[y == c]

            for j, feature in enumerate(samples):

                m = np.mean(feature)
                n = len(feature)
                s = np.std(feature, ddof=1)

                self.likelihood_.append(dict(mean=m, std=s, n=n))

        return self.likelihood_

    def _update_likelihood(self, X, y):

        likelihoods = {}

        for i, c in enumerate(self.classes_):
            feature_probas = []

            for i, x in enumerate(X.transpose()):

                x = x[y == c]

                old_m = self.likelihood_[c][i]["mean"]
                old_std = self.likelihood_[c][i]["std"]
                old_n = self.likelihood_[c][i]["n"]

                n = old_n + len(x)

                m = (old_m * old_n + np.mean(x) * n) / (old_n + n)

                s = np.sqrt((old_n * (old_std**2 + (old_m - m)**2) + len(x) * (np.var(x) + (np.mean(x) - m)**2))
                            / (old_n + len(x)))

                self.likelihood_[c][i] = dict(mean=m, std=s, n=n)

        return self.likelihood_

    def _get_likelihood(self, sample, c):

        likelihood = 1

        for i, feature in enumerate(sample):
            mean = self.likelihood_[c][i]["mean"]
            std = self.likelihood_[c][i]["std"]

            likelihood *= self._pdf(feature, mean, std)

        return likelihood


class BernoulliNB(NBClassifier):

    def _pdf(self, x, p):

        return (1 - x) * (1 - p) + x * p

    def _fit_evidence(self, X):

        feature_probas = []

        for feature in X.transpose():

            count = len(feature == 1)
            n = len(feature)

            feature_probas.append(dict(count=count, n=n))

        return np.array(feature_probas)

    def _get_evidence(self, X):

        evidence = []

        for i, feature in enumerate(X.transpose()):

            count = self.evidence_[i]["count"]
            n = self.evidence_[i]["n"]

            evidence.append(self._pdf(feature, count / n))

        return np.array(evidence).transpose()

    def _update_evidence(self, X):

        for i, feature in enumerate(X.transpose()):

            count = old_count + len(feature == 1)
            n = old_n + len(feature)

            self.evidence_[i] = dict(count=count, n=n)

        return self.evidence_

    def _fit_likelihood(self, X, y):

        return {c: self._fit_evidence(X[y == c]) for c in self.classes_}

    def _update_likelihood(self, X, y):

        likelihoods = {}

        for i, c in enumerate(self.classes_):
            feature_probas = []

            for i, x in enumerate(X.transpose()):

                x = x[y == c]

                count = old_count + len(feature == 1)
                n = old_n + len(feature)

                self.likelihood_[c][i] = dict(count=count, n=n)

        return self.likelihood_

    def _get_likelihood(self, sample, c):

        likelihood = 1

        for i, feature in enumerate(sample):
            count = self.likelihood_[c][i]["count"]
            n = self.likelihood_[c][i]["n"]

            likelihood *= self._pdf(feature, count / n)

        return likelihood


class MultinomialNB(NBClassifier):

    def _pdf(self, x, mean, std):

        num = np.exp(-((x - mean)**2) / (2 * std**2))
        den = np.sqrt(2 * np.pi * std**2)

        return num / den

    def _fit_evidence(self, X):

        feature_probas = []

        for feature in X.transpose():

            m = np.mean(feature)
            s = np.std(feature, ddof=0)
            n = len(feature)

            feature_probas.append(dict(mean=m, std=s, n=n))

        return np.array(feature_probas)

    def _get_evidence(self, X):

        evidence = []

        for i, feature in enumerate(X.transpose()):

            mean = self.evidence_[i]["mean"]
            std = self.evidence_[i]["std"]

            evidence.append(self._pdf(feature, mean, std))

        return np.array(evidence).transpose()

    def _update_evidence(self, X):

        for i, feature in enumerate(X.transpose()):

            old_m = self.evidence_[i]["mean"]
            old_std = self.evidence_[i]["std"]
            old_n = self.evidence_[i]["n"]

            n = old_n + len(feature)

            m = (old_m * old_n + np.mean(feature) * n) / (old_n + n)

            s = np.sqrt((old_n * (old_std**2 + (old_m - m)**2) + len(feature) *
                         (np.var(feature) + (np.mean(feature) - m)**2)) / (old_n + len(feature)))

            self.evidence_[i] = dict(mean=m, std=s, n=n)

        return self.evidence_

    def _fit_likelihood(self, X, y):

        return {c: self._fit_evidence(X[y == c]) for c in self.classes_}

    def _update_likelihood(self, X, y):

        likelihoods = {}

        for i, c in enumerate(self.classes_):
            feature_probas = []

            for i, x in enumerate(X.transpose()):

                x = x[y == c]

                old_m = self.likelihood_[c][i]["mean"]
                old_std = self.likelihood_[c][i]["std"]
                old_n = self.likelihood_[c][i]["n"]

                n = old_n + len(x)

                m = (old_m * old_n + np.mean(x) * n) / (old_n + n)

                s = np.sqrt((old_n * (old_std**2 + (old_m - m)**2) + len(x) * (np.var(x) + (np.mean(x) - m)**2))
                            / (old_n + len(x)))

                self.likelihood_[c][i] = dict(mean=m, std=s, n=n)

        return self.likelihood_

    def _get_likelihood(self, sample, c):

        likelihood = 1

        for i, feature in enumerate(sample):
            mean = self.likelihood_[c][i]["mean"]
            std = self.likelihood_[c][i]["std"]

            likelihood *= self._pdf(feature, mean, std)

        return likelihood
