import numpy as np

from .components.data import Data
from .components.modules import MatMul
from .components.modules.activation_functions import Relu
from .components.modules.loss_functions import MSE
from .components.net import Net
from .utils.grad_check import gradient_checker


def f(X):
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 4
factor = 5
X_train = np.random.rand(m_train, 3) * factor
Y_train = f(X_train)


# Construct neural network
ActF = Relu
LossF = MSE

x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

z = Data(ID = 'z', shape = (1, 1))
matmul = MatMul(ID = 'ActF', inputs = inputs, output = z)

# Initialize target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = ())
sum_loss = LossF(ID = 'LossF', inputs = [z, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set required values
x.val = X_train
y.val = Y_train

# Run network forward pass
net.forward()


# Perform gradient checking
gradient_checker(net = net, data_obj = matmul.W, loss_obj = loss, h = 1e-5)