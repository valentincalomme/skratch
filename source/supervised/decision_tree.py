import numpy as np

from utils.evaluation import entropy, gini


class DecisionNode:

    def __init__(self):

        pass


class DecisionTree:

    def __init__(self, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, seed=0):

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rnd = np.random.RandomState(seed)

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        self.root = DecisionNode()

        return self

    def _evaluate_split(self, left, right):

        

    def 

    def predict(self, X):

        return None
