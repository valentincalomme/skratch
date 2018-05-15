import numpy as np

from utils.distances import correlation


def accuracy(y, yhat):
    return np.mean(y == yhat)


def mse(yhat, y):
    return np.mean((y - yhat) ** 2)


def rmse(yhat, y):
    return np.sqrt(MSE(y, yhat))


def mae(yhat, y):
    return np.mean(abs(y - yhat))


def msle(yhat, y):
    return np.mean(np.square(np.log(y + np.ones_like(y)) - np.log(yhat + np.ones_like(yhat))))


def med_a_e(yhat, y):
    return np.median(np.abs(y - yhat))


def rsquared(yhat, y):
    return 1 - (np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2))


def brier_score(f, o):
    return MSE(f, o)


def entropy(x):

    unique, counts = np.unique(x, return_counts=True)

    frequencies = counts / np.sum(counts)

    return - np.sum(frequencies * np.log2(frequencies))


def confusion_matrix(output, target):

    matrix = {}

    # Extract the classes
    u1 = np.unique(output)
    u2 = np.unique(target)

    # Populates matrix
    for o in u1:
        for t in u2:
            matrix[(o, t)] = np.sum((target == t) & (output == o))

    return matrix


def recall(output, target):

    result = {}
    unique, counts = np.unique(target, return_counts=True)

    for u in unique:
        result[u] = np.sum((target == u) & (output == u)) / np.sum(target == u)

    weighted_recall = np.sum([result[unique[u]] * counts[u] / len(target) for u in range(len(unique))])

    return weighted_recall, result


def precision(output, target):

    result = {}
    unique, counts = np.unique(target, return_counts=True)

    for u in unique:
        result[u] = np.sum((target == u) & (output == u)) / np.sum(output == u)

    weighted_precision = np.sum([result[unique[u]] * counts[u] / len(target) for u in range(len(unique))])

    return weighted_precision, result


def f1_score(output, target):

    result = {}
    unique, counts = np.unique(target, return_counts=True)

    # Computes the recall and precision of these classes
    _, p = precision(output, target)
    _, r = recall(output, target)

    # Computes the F1 score as the harmonic mean between precision and recall
    for u in unique:
        result[u] = 2 * (p[u] * r[u]) / (p[u] + r[u])

    weighted_F1_score = np.sum([result[unique[u]] * counts[u] / len(target) for u in range(len(unique))])

    return weighted_F1_score, result
