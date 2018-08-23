from utils.optimization import StochasticGradientDescentOptimizer, NesterovAcceleratedGradientOptimizer
from utils.regularization import *
from utils.evaluation import mse, accuracy

optimizers = [StochasticGradientDescentOptimizer(learning_rate=0.01, momentum=0.3),
              NesterovAcceleratedGradientOptimizer(learning_rate=0.01, momentum=0.3)]

regularizers = [BaseRegularizer(),
                LASSO(_lambda=0.2),
                Ridge(_lambda=0.2),
                ElasticNet(_lambda=0.2)]

n_samples = 10000
n_features = 20
maximum = 1
minimum = -1

X = np.random.rand(n_samples, n_features)
weights = (maximum - minimum) * np.random.rand(n_features + 1) + minimum
y = add_dummy_feature(X).dot(weights) > 0

for optimizer in optimizers:
    for regularizer in regularizers:

        print(optimizer.__class__.__name__, regularizer.__class__.__name__)

        reg = LogisticRegression(optimizer, regularizer=regularizer)

        for weights_, loss in reg._fit(X, y):

            reg.coef_ = weights_

        print(accuracy(y, reg.predict(X)))
