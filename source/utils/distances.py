import numpy as np
import collections

def validate_input(x, y):
    """
    Ensures that both x and y are numpy arrays or objects that can be transformed
    into numpy arrays. It also ensures that both arrays are the same size
    """

    x = np.array(x) # makes sure x is a numpy array
    y = np.array(y) # makes sure y is a numpy array

    assert len(x) == len(y), "both vectors must have the same length"

    return x, y

def MinkowskiDistance(x, y, n):
    """
    Computes the Minkowski distance between two points
    :param x: first point
    :param y: second point
    :param n: power
    :return: Minkowski distance between x and y
    """

    x, y = validate_input(x, y)

    # Ensures that the power is valid
    assert n >= 1, 'power must be greater or equal to 1'

    return (np.sum(np.abs(x - y) ** n)) ** (1 / n)

def EuclideanDistance(x, y):
    """
    Computes the euclidean distance between two points
    :param x: first point
    :param y: second point
    :return: euclidean distance between x and y
    """
    return MinkowskiDistance(x, y, 2)


def ManhattanDistance(x, y):
    """
    Computes the Manhattan distance between two points
    :param x: first point
    :param y: second point
    :return: Manhattan distance between x and y
    """
    return MinkowskiDistance(x, y, 1)


def CanberraDistance(x, y):
    """
     Computes the Canberra distance between two points
     :param x: first point
     :param y: second point
     :return: Canberra distance between x and y
    """

    x, y = validate_input(x, y)

    return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))


def CosineSimilarity(x, y):
    """
     Computes the cosine similarity between two points
     :param x: first point
     :param y: second point
     :return: cosine similarity  between x and y
    """

    x, y = validate_input(x, y)

    return np.sum(x * y) / ((np.sqrt(np.sum(x ** 2))) * (np.sqrt(np.sum(y ** 2))))


def PearsonCorrelation(x, y):
    """
    Computes the Pearson Correlation between two variables
    :param x: variable x
    :param y: variable y
    :return: Pearson Correlation between x and y
    """

    x, y = validate_input(x, y)

    return np.corrcoef(x, y)[0, 1]


def HammingDistance(x:str, y:str):
    """
    Computes the hamming distance between two strings/points
    :param x: first string/point
    :param y: first string/point
    :return: hamming distance between x and y
    """
    
    # Ensures that both vectors are numpy arrays
    x = np.array([x for x in x])
    y = np.array([y for y in y])

    # Ensures that both vectors are the same length
    assert len(x) == len(y), 'both vectors must have the same length'

    return np.sum(x != y)


def ChebyshevDistance(x, y):
    """
    Computes the Chebyshev distance between two points
    :param x: first point
    :param y: first point
    :return: Chebyshev distance between x and y
    """

    x, y = validate_input(x, y)

    return np.max(np.abs(x - y))


def LevenshteinDistance(x, y):
    """
    Computes the Levenhstein/Edit distance between two strings x and y
    :param x: first string
    :param y: second string
    :return: Levenhstein/Edit distance between x and y
    """

    # Computes matrix
    matrix = np.zeros((len(x) + 1, len(y) + 1))
    matrix[:, 0] = np.array([i for i in range(len(x) + 1)])
    matrix[0, :] = np.array([i for i in range(len(y) + 1)])

    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            if x[i - 1] == y[j - 1]:
                matrix[i, j] = matrix[i - 1, j - 1]
            else:
                matrix[i, j] = min(matrix[i - 1, j - 1], matrix[i - 1, j], matrix[i, j - 1]) + 1
                
    return matrix[len(x), len(y)]


def EditDistance(x, y):
    """
    Computes the Levenhstein/Edit distance between two strings x and y
    :param x: first string
    :param y: second string
    :return: Levenhstein/Edit distance between x and y
    """
    return LevenshteinDistance(x, y)


def DamerauLevenshteinDistance(x, y):
    """
    Computes the Damerau-Levenshtein distance between two strings x and y
    :param x: first string
    :param y: second string
    :return: Damerau-Levenshtein distance between x and y
    """

    # Computes matrix
    matrix = np.zeros((len(x) + 1, len(y) + 1))
    matrix[:, 0] = np.array([i for i in range(len(x) + 1)])
    matrix[0, :] = np.array([i for i in range(len(y) + 1)])

    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            cost = 1 - (x[i - 1] == y[j - 1])

            matrix[i, j] = min(matrix[i - 1, j] + 1, matrix[i, j - 1] + 1, matrix[i - 1, j - 1] + cost)

            if i > 1 and j > 1:
                if x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                    matrix[i, j] = min(matrix[i, j], matrix[i - 2, j - 2] + cost)

    return matrix[len(x) - 1, len(y) - 1]
