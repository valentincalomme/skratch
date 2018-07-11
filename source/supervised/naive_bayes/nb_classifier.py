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
        self.likelihood_ = self._fit_likelihood(X, y)
        self.evidence_ = self._fit_evidence(X)

        return self

    def update(self, X, y):

        self.priors_ = self._update_priors(y)
        self.likelihood_ = self._update_likelihood(X, y)
        self.evidence_ = self._update_evidence(X)

        return self

    def _fit_prior(self, y):

        self.classes_, self.priors_ = np.unique(y, return_counts=True)

        return self.priors_

    def _update_priors(self, y):

        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ += counts

        return self.priors_

    def _get_prior(self, c):

        return self.priors_[c] / np.sum(self.priors_)
