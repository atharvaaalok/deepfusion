import numpy as np

from .components.data import Data
from .components.modules import MatMul
from .components.modules.activation_functions import Relu
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


# Construct neural network
x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

z1 = Data(ID = 'z', shape = (1, 10))
matmul = MatMul(ID = 'Matmul', inputs = inputs, output = z1)

a = Data(ID = 'a', shape = (1, 10))
relu = Relu(ID = 'Relu', inputs = [z1], output = a)

z2 = Data(ID = 'z2', shape = (1, 1))
matmul2 = MatMul(ID = 'Matmul2', inputs = [a], output = z2)

# Attach loss function and variable
loss1 = Data(ID = 'loss1', shape = (1, 1))
sum_loss1 = SumLoss(ID = 'SumLoss1', inputs = [z2], output = loss1)

# Attach another loss function and variable
loss2 = Data(ID = 'loss2', shape = (1, 1))
sum_loss2 = SumLoss(ID = 'SumLoss2', inputs = [z2], output = loss2)


# Initialize the network
net = Net(ID = 'net', root_nodes = [loss1, loss2])

# # Visualize the network
# net.visualize()
# print([ele.ID for ele in net.topological_order])


# Set required values
x.val = X_train

net.forward()
net.backward()

print(x.deriv)