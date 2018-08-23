from utils.optimization import StochasticGradientDescentOptimizer, NesterovAcceleratedGradientOptimizer
from utils.regularization import *
from utils.evaluation import mse

optimizers = [StochasticGradientDescentOptimizer(learning_rate=0.01, momentum=0.1),
              NesterovAcceleratedGradientOptimizer(learning_rate=0.01, momentum=0.1)]

regularizers = [BaseRegularizer(),
                LASSO(_lambda=0.2),
                Ridge(_lambda=0.2),
                ElasticNet(_lambda=0.2)]

num_samples = 1000
num_features = 20

np.random.seed(0)

X = np.random.rand(num_samples, num_features)
weights = np.random.rand(X.shape[1] + 1)
y = add_dummy_feature(X).dot(weights)

for optimizer in optimizers:
    for regularizer in regularizers:

        print(optimizer.__class__.__name__, regularizer.__class__.__name__)

        reg = LinearRegression(optimizer, regularizer=regularizer)

        reg.fit(X, y)

        print(mse(y, reg.predict(X)))
