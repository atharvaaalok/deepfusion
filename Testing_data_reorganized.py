import numpy as np

from .components.data import Data
from .components.modules.activation_functions import Relu
from .components.net import Net


def f(X):
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 10
factor = 5
X_train = np.random.rand(m_train, 3) * factor
Y_train = f(X_train)


# Construct neural network
ActF = Relu

x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

z = Data(ID = 'z', shape = (1, 3))
act_f = ActF(ID = 'ActF', inputs = inputs, output = z)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [z])


# Set required values
x.val = X_train

# Run network forward pass
net.forward()

print(x.val)
print(z.val)