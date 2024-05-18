import numpy as np

from .components.data import Data
from .components.modules.rnn import RNN1
from .components.modules.rnn import RNN2
from .components.modules.loss_functions import SumLoss
from .components.net import Net
from .utils.grad_check import gradient_checker


def f(X):
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Construct a 1 unit RNN
m_train = 4
a_train = np.zeros((m_train, 3))
factor = 5
X_train = np.random.rand(m_train, 3) * factor
Y_train = f(X_train)


B = m_train

# Construct the neural network
a0 = Data(ID = 'a0', shape = (B, 3))
x0 = Data(ID = 'x0', shape = (B, 3))

a1 = Data(ID = 'a1', shape = (B, 3))
rnn1 = RNN1(ID = 'RNN1', inputs = [a0, x0], output = a1)

y1 = Data(ID = 'y1', shape = (B, 1))
rnn2 = RNN2(ID = 'RNN2', inputs = [a1], output = y1)

# Attach loss layer
loss = Data(ID = 'loss', shape = (1, 1))
mse = SumLoss(ID = 'Sumloss', inputs = [y1], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# # Visualize the network
net.visualize()


# Set required values
a0.val = a_train
x0.val = X_train


gradient_checker(net = net, data_obj = x0, h = 1e-6)