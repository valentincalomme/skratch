# ML FROM SCRATCH #

def linear_kernel(x1, x2):

    return np.inner(x1, x2)


def polynomial_kernel(x1, x2, power, coef):

    return (np.inner(x1, x2) + coef)**power


def rbf_kernel(x1, x2, gamma):

    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * distance)
