import numpy as np

from ..components.data import Data
from ..components.modules import MatMul
from ..components.modules.activation_functions import Relu
from ..components.modules.normalizations import LayerNorm
from ..components.modules.loss_functions import SumLoss
from ..components.net import Net
from ..utils.grad_check import gradient_checker


def f(X):
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training and test data
m_train = 4

# Generate input data
factor = 5
X_train = np.random.rand(m_train, 3) * factor
Y_train = f(X_train)


# Construct neural network
x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

z = Data(ID = 'z', shape = (1, 10))
matmul = MatMul(ID = 'Matmul', inputs = inputs, output = z)

z_norm = Data(ID = 'z_norm', shape = (1, 10))
norm = LayerNorm(ID = 'Norm', inputs = [z], output = z_norm)

a = Data(ID = 'a', shape = (1, 10))
relu = Relu(ID = 'relu', inputs = [z_norm], output = a)

# Initialize, loss variable and attach loss function
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = SumLoss(ID = 'SumLoss', inputs = [a], output = loss)

# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set required values
x.val = X_train


gradient_checker(net = net, data_obj = x, loss_obj = loss)