import numpy as np


class Optimizer(object):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def update(self, x):
        raise NotImplementedError

    def gen_optimization_steps(self, x):
        raise NotImplementedError

    def gradient_numerical_estimation(self, x, with_respect_to=None):
        input_type = type(x)

        if input_type != np.ndarray:
            x = np.array([x])

        if with_respect_to is None:
            with_respect_to = range(len(x))

        gradients = np.zeros_like(x, dtype=float)
        compute_gradient = lambda x, h: (self.function(x + h) - self.function(x - h)) / (2 * np.sum(h))

        for i in with_respect_to:
            h = np.zeros_like(x, dtype=float)
            h[i] = 1E-8
            gradients[i] = compute_gradient(x, h)

        if input_type != np.ndarray:
            gradients = np.squeeze(gradients)

        return gradients

    def complete_optimization(self, x):
        for i in self.gen_optimization_steps(x):
            result = i

        return result


class GradientDescentOptimizer(Optimizer):
    def __init__(self, function, gradient=None, learning_rate=0.2, stopping_criterion=1E-6):
        self.function = function
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.stopping_criterion = stopping_criterion

        if gradient is None:
            self.gradient = self.gradient_numerical_estimation

    def do_optimization_step(self, x):
        return x - self.learning_rate * self.gradient(x)

    def gen_optimization_steps(self, x):
        last_value = float('inf')
        current_value = 0.0

        while (last_value - current_value > self.stopping_criterion):
            last_value = self.function(x)
            x = self.do_optimization_step(x)
            current_value = self.function(x)
            yield x


class CoordinateDescent(Optimizer):
    def __init__(self, function, gradient=None, alpha=0.2, epsilon=1E-6):
        self.function = function
        self.gradient = gradient
        self.alpha = alpha
        self.epsilon = epsilon

        if not gradient:
            self.gradient = self.gradient_estimation

    def update(self, x, with_respect_to):
        return x - self.alpha * self.g(x, with_respect_to)

    def gen(self, x):

        def improvable_directions(current_weights):
            """ Detect which direction would minimize the cost function """
            return [(self.f(current_weights) - self.f(self.update(current_weights, with_respect_to=i))) > self.epsilon
                    for i in range(len(current_weights))]

        possible_to_improve = improvable_directions(x)
        while any(possible_to_improve):
            direction = np.where(possible_to_improve)[0][0]

            last_value = float('inf')
            current_value = 0.0

            while (last_value - current_value > self.epsilon):
                last_value = self.f(x)
                x = self.update(x, with_respect_to=direction)
                current_value = self.f(x)
                yield x

            possible_to_improve = improvable_directions(x)


class StochasticGradientDescent(Optimizer):
    pass


class Adagrad(Optimizer):
    pass


class Adadelta(Optimizer):
    pass


class RMSprop(Optimizer):
    pass


class Adam(Optimizer):
    pass


class NesterovAcceleratedGradient(Optimizer):
    pass
