import numpy as np

from .components.data import Data
from .components.modules import MatMul
from .components.modules.activation_functions import Tanh
from .components.modules.loss_functions import MSE
from .components.net import Net
from .utils.grad_check import gradient_checker


def f(X):
    Y = X[0, :] + 2 * X[1, :] ** 2 + 3 * X[2, :] ** 0.5
    return Y.reshape(1, -1)


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 3
factor = 5
X_train = np.random.rand(3, m_train) * factor
Y_train = f(X_train)


# Construct neural network
x = Data(ID = 'x', shape = (3, 1))
inputs = [x]

z = Data(ID = f'z', shape = (10, 1))
matmul = MatMul(ID = f'MatMul', inputs = inputs, output = z, weight_init_type = 'Random')

a = Data(ID = f'a', shape = (10, 1))
tanh = Tanh(ID = f'Tanh', inputs = [z], output = a)

z2 = Data(ID = f'z2', shape = (1, 1))
mat = MatMul(ID = f'MatMul2', inputs = [a], output = z2, weight_init_type = 'Random')

# Initialize target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
mse = MSE(ID = 'MSE', inputs = [z2, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set required values
x.val = X_train
y.val = Y_train


gradient_checker(net = net, data_obj = z, loss_obj = loss, h = 1e-5)