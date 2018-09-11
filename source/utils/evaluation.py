import numpy as np

from utils.distances import correlation, euclidean, cdist
from itertools import combinations


def silhouette_index(X, y, distance=euclidean):

    def inter_cluster_distance(sample_id):

        other_sample_ids = np.array([i for i in range(len(X)) if i != sample_id])
        cluster_ids = y[other_sample_ids]
        cluster_id = y[sample_id]
        x = X[sample_id]

        cluster = X[other_sample_ids][cluster_ids == cluster_id]

        if len(cluster) == 0:
            return 0
        else:
            distances = cdist([x], cluster, distance)
            return np.mean(distances)

    def nearest_cluster_distance(sample_id):

        x = X[sample_id]
        dist = float('inf')

        for other_cluster_id in set(y[y != y[sample_id]]):

            cluster = X[y == other_cluster_id]

            new_dist = np.mean(cdist([x], cluster, distance))

            dist = new_dist if new_dist < dist else dist

        return dist

    s = []

    for i, x in enumerate(X):

        a = inter_cluster_distance(i)
        b = nearest_cluster_distance(i)

        s.append((b - a) / max(a, b))

    return np.mean(s)

def rand_index(X, y):

    a = b = c = d = 0

    for i, j in combinations(range(len(X)), 2):

        a += X[i] == X[j] and y[i] == y[j]
        b += X[i] != X[j] and y[i] != y[j]
        c += X[i] == X[j] and y[i] != y[j]
        d += X[i] != X[j] and y[i] == y[j]

    return (a + b) / (a + b + c + d)


def accuracy(predicted_target, target):
    correctly_predicted = target == predicted_target

    return np.mean(correctly_predicted)


def mse(predicted_target, target):
    errors = target - predicted_target

    return np.mean(errors ** 2)


def rmse(predicted_target, target):
    mse = MSE(target, predicted_target)

    return np.sqrt(mse)


def mae(predicted_target, target):
    errors = target - predicted_target

    return np.mean(abs(errors))


def msle(predicted_target, target):
    log_target = np.log(target + np.ones_like(target))
    log_predicted_target = np.log(predicted_target + np.ones_like(predicted_target))

    log_errors = log_target - log_predicted_target

    return np.mean(np.square(log_errors))


def med_a_e(predicted_target, target):
    errors = target - predicted_target

    return np.median(np.abs(errors))


def rsquared(predicted_target, target):
    errors = target - predicted_target

    return 1 - (np.sum((errors)**2) / np.sum((target - np.mean(target))**2))


def brier_score(f, o):

    return MSE(f, o)


def gini(x):

    min_value = np.min(x)

    x -= min_value * (min_value < 0)  # makes sure input is positive
    x += 1E-10  # values cannot be zero
    x = np.sort(x)  # values must be sorted

    return np.sum((2 * np.arange(1, 1 + len(x)) - len(x) - 1) * x) / (len(x) * np.sum(x))


def entropy(x):

    unique, counts = np.unique(x, return_counts=True)

    frequencies = counts / np.sum(counts)

    return - np.sum(frequencies * np.log2(frequencies))


def confusion_matrix(predicted_target, target):

    confusion_matrix = {}

    # Extract the classes
    predicted_target_classes = np.unique(predicted_target)
    target_classes = np.unique(target)

    # Populates confusion matrix
    for t in target_classes:
        confusion_matrix[t] = {o: np.sum((target == t) & (predicted_target == o)) for o in predicted_target_classes}

    return confusion_matrix


def print_confusion_matrix(confusion_matrix):

    for (o, t), v in confusion_matrix.items():


def recall(predicted_target, target):

    recall_per_class = {}
    classes, counts = np.unique(target, return_counts=True)

    for c in classes:

        tp = (target == c) & (predicted_target == c)
        fn = (target == c) & (predicted_target != c)

        recall_per_class[c] = np.sum(tp) / np.sum(tp + fn)

    weighted_recall = np.sum([recall_per_class[c] * counts[c] / len(target) for c in classes])

    return weighted_recall, recall_per_class


def precision(predicted_target, target):

    precision_per_class = {}
    classes, counts = np.unique(target, return_counts=True)

    for c in classes:

        tp = (target == c) & (predicted_target == c)
        fp = (target != c) & (predicted_target == c)

        precision_per_class[c] = np.sum(tp) / np.sum(tp + fp)

    weighted_precision = np.sum([precision_per_class[c] * counts[c] / len(target) for c in classes])

    return weighted_precision, precision_per_class


def fbeta_score(predicted_target, target, beta=1.0):

    fbeta_score_per_class = {}
    classes, counts = np.unique(target, return_counts=True)

    # Computes the recall and precision of these classes
    _, p = precision(predicted_target, target)
    _, r = recall(predicted_target, target)

    # Computes the F-beta score as the harmonic mean between precision and recall
    for c in classes:
        if beta**2 * p[c] + r[c] == 0:
            fbeta_score_per_class[c] = 0  # if precision and recall are 0, then f-beta should also be zero
        else:
            fbeta_score_per_class[c] = (1 + beta**2) * (p[c] * r[c]) / (beta**2 * p[c] + r[c])

    weighted_fbeta_score = np.sum([fbeta_score_per_class[c] * counts[c] / len(target) for c in classes])

    return weighted_fbeta_score, fbeta_score_per_class


def f1_score(predicted_target, target):

    return fbeta_score(predicted_target, target, beta=1.0)
