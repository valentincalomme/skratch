from itertools import product

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


import collections


def pdist(samples, distance):

    distances = np.empty((len(samples), len(samples)))

    for i, a in enumerate(samples):
        for j, b in enumerate(samples[i:], i):

            dist = distance(a, b)

            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def cdist(A, B, distance):

    distances = np.empty((len(A), len(B)))

    for i, a in enumerate(A):
        for j, b in enumerate(B):

            dist = distance(a, b)

            distances[i, j] = dist

    return distances


def minkowski(x, y, n):

    if n < 1 or n % 1 != 0:
        raise ValueError("n should be a strictly positive integer, not {}".format(n))

    return (np.sum(np.abs(x - y) ** n)) ** (1 / n)


def euclidean(x, y):

    return minkowski(x, y, 2)


def sqeuclidean(x, y):

    return np.sum(np.abs(x - y)**2.0)


def manhattan(x, y):

    return minkowski(x, y, 1)


def canberra(x, y):

    return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))


def braycurtis(x, y):

    return np.sum(np.abs(x - y) / np.abs(x + y))


def cosine(x, y):

    return 1.0 - (np.sum(x * y) / ((np.sqrt(np.sum(x ** 2.0))) * (np.sqrt(np.sum(y ** 2.0)))))


def correlation(x, y):

    return 1.0 - np.corrcoef(x, y)[0, 1]


def hamming(x: str, y: str):

    # Ensures that both vectors are numpy arrays
    x = np.array([x for x in x])
    y = np.array([y for y in y])

    return np.sum(x != y)


def chebyshev(x, y):

    return np.max(np.abs(x - y))


def dice(x, y):

    ctf = np.sum(x * np.logical_not(y))
    cft = np.sum(y * np.logical_not(x))
    ctt = np.sum(x * y)

    if (2.0 * ctt + cft + ctf) == 0:
        raise ZeroDivisionError()

    return (ctf + cft) / (2.0 * ctt + cft + ctf)


def kulsinski(x, y):

    ctf = np.sum(x * np.logical_not(y))
    cft = np.sum(y * np.logical_not(x))
    ctt = np.sum(x * y)
    n = len(x)

    if (cft + ctf + n) == 0:
        raise ZeroDivisionError()

    return (ctf + cft - ctt + n) / (cft + ctf + n)


def jaccard(x, y):

    ctf = np.sum(x * np.logical_not(y))
    cft = np.sum(y * np.logical_not(x))
    ctt = np.sum(x * y)

    if (ctt + cft + ctf) == 0:
        raise ZeroDivisionError()

    return (ctf + cft) / (ctt + cft + ctf)


def rogerstanimoto(x, y):

    ctf = np.sum(x * np.logical_not(y))
    cft = np.sum(y * np.logical_not(x))
    ctt = np.sum(x * y)
    cff = np.sum(np.logical_not(x) * np.logical_not(y))

    R = 2.0 * (ctf + cft)

    if (ctt + cff + R) == 0:
        raise ZeroDivisionError()

    return R / (ctt + cff + R)


def sokalmichener(x, y):

    ctf = np.sum(x * np.logical_not(y))
    cft = np.sum(y * np.logical_not(x))
    ctt = np.sum(x * y)
    cff = np.sum(np.logical_not(x) * np.logical_not(y))

    R = 2.0 * (ctf + cft)
    S = (cff + ctt)

    if (S + R) == 0:
        raise ZeroDivisionError()

    return R / (S + R)


def sokalsneath(x, y):

    ctf = np.sum(x * np.logical_not(y))
    cft = np.sum(y * np.logical_not(x))
    ctt = np.sum(x * y)

    R = 2.0 * (ctf + cft)

    if (ctt + R) == 0:
        raise ZeroDivisionError()

    return R / (ctt + R)


def yule(x, y):

    ctf = np.sum(x * np.logical_not(y))
    cft = np.sum(y * np.logical_not(x))
    ctt = np.sum(x * y)
    cff = np.sum(np.logical_not(x) * np.logical_not(y))

    R = 2.0 * (ctf * cft)

    if (ctt * cff + R / 2.0) == 0:
        raise ZeroDivisionError()

    return R / (ctt * cff + R / 2.0)


def russellrao(x, y):

    ctt = np.sum(x * y)
    n = len(x)

    return (n - ctt) / n


def levenshtein(x, y):

    # Computes matrix
    matrix = np.zeros((len(x) + 1, len(y) + 1))
    matrix[:, 0] = np.array([i for i in range(len(x) + 1)])
    matrix[0, :] = np.array([i for i in range(len(y) + 1)])

    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            
            cost = 1 - (x[i - 1] == y[j - 1])

            matrix[i, j] = min(matrix[i - 1, j - 1] + cost,
                               matrix[i - 1, j] + 1,
                               matrix[i, j - 1] + 1)


    return matrix[len(x), len(y)]


def restricted_levenshtein(x, y):

    # Computes matrix
    matrix = np.zeros((len(x) + 1, len(y) + 1))
    matrix[:, 0] = np.array([i for i in range(len(x) + 1)])
    matrix[0, :] = np.array([i for i in range(len(y) + 1)])

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):

            cost = 1 - (x[i - 1] == y[j - 1])

            matrix[i, j] = min(matrix[i - 1, j - 1] + cost,
                               matrix[i - 1, j] + 1,
                               matrix[i, j - 1] + 1)

            if i > 1 and j > 1:
                if x[i -1] == y[j - 2] and x[i - 2] == y[j - 1]:
                    matrix[i, j] = min(matrix[i, j],
                                       matrix[i - 2, j - 2] + cost)

    return matrix[len(x) - 1, len(y) - 1]
