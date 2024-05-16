import numpy as np


def generate_data(m_train):

    # Generate input and target data
    factor = 5
    X_train = np.random.rand(m_train, 3) * factor
    Y_train = f(X_train)

    return X_train, Y_train


def f(X):
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y