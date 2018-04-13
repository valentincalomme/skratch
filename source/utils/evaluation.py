import numpy as np

from utils.distances import PearsonCorrelation

def validate_input(y, yhat):
    """
    Ensures that both arguments are numpy arrays of the same
    size
    """

    y, yhat = validate_input(y, yhat)

    return y, yhat

def accuracy(y, yhat):
    """
    Computes accuracy of a class prediction based on the target and output class
    :param y: target class
    :param yhat: outputted class
    :return: accuracy (between 0 and 1 inclusive)
    """

    y, yhat = validate_input(y, yhat)

    return np.mean(y == yhat)


def MSE(yhat, y):
    """
    Computes mean-squared error between a target variable and an estimate
    :param y: target variable
    :param yhat: estimate
    :return: mean-squared error (0 is best)
    """

    y, yhat = validate_input(y, yhat)

    return np.mean((y - yhat) ** 2)


def RMSE(yhat, y):
    """
    Computes root mean-squared error between a target variable and an estimate
    :param y: target variable
    :param yhat: estimate
    :return: root mean-squared error (0 is best)
    """
    return np.sqrt(MSE(y, yhat))


def MAE(yhat, y):
    """
    Computes mean absolute error between a target variable and an estimate
    :param y: target variable
    :param yhat: estimate
    :return: mean absolute error  (0 is best)
    """

    y, yhat = validate_input(y, yhat)

    return np.mean(abs(y - yhat))

def MSLE(yhat, y):
    """
    Computes mean-squared logarithmic error between a target variable and an estimate
    :param y: target variable
    :param yhat: estimate
    :return: mean-squared logarithmic error
    """

    y, yhat = validate_input(y, yhat)

    return np.mean(np.square(np.log(y + np.ones_like(y)) - np.log(yhat + np.ones_like(yhat))))


def MedAE(yhat, y):
    """
    Computes median absolute error between a target variable and an estimate
    :param y: target variable
    :param yhat: estimate
    :return: median absolute error
    """

    y, yhat = validate_input(y, yhat)

    return np.median(np.abs(y - yhat))


def Rsquared(yhat, y):
    """
    Computes the coefficient of determination between a target variable and an estimate
    :param y: target variable
    :param yhat: estimate
    :return: coefficient of determination ("R squared")
    """

    y, yhat = validate_input(y, yhat)

    return  1 - (np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2))


def BrierScore(f, o):
    """
    Computes Brier's score
    :param f: forecasted probability
    :param o: observed probability
    :return: Brier's score
    """

    f, o = validate_input(f, o)

    return MSE(f, o)


def confusion_matrix(output, target):
    """
    Computes the confusion matrix based on a target class and an output class
    :param output: class that is outputted
    :param target: true class
    :return: confusion matrix ([{output,target}:frequency])
    """

    target, output = validate_input(target, output)

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
    """
    Computes the recall between output and target classes
    :param output: output class
    :param target: target class
    :return: recall for each class
    """

    target, output = validate_input(target, output)

    result = {}
    unique, counts = np.unique(target, return_counts=True)

    for u in unique:
        result[u] = np.sum((target == u) & (output == u)) / np.sum(target == u)

    weighted_recall = np.sum([result[unique[u]] * counts[u] / len(target) for u in range(len(unique))])

    return weighted_recall, result


def precision(output, target):
    """
    Computes the precision between output and target classes
    :param output: output class
    :param target: target class
    :return: precision for each class
    """

    target, output = validate_input(target, output)


    result = {}
    unique, counts = np.unique(target, return_counts=True)

    for u in unique:
        result[u] = np.sum((target == u) & (output == u)) / np.sum(output == u)

    weighted_precision = np.sum([result[unique[u]] * counts[u] / len(target) for u in range(len(unique))])

    return weighted_precision, result


def f1_score(output, target):
    """
    Computes the F1 score between output and target classes
    :param output: output class
    :param target: target class
    :return: f1 score for each class
    """

    
    target, output = validate_input(target, output)

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


def entropy(x):
    """
    Computes the entropy/purity of a discrete variable
    :param x: discrete variable
    :return: entropy of the discrete variable
    """

    unique, counts = np.unique(x, return_counts=True)

    frequencies = counts / np.sum(counts)
    
    return - np.sum(frequencies * np.log2(frequencies))
