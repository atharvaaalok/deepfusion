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
a_train = np.random.rand(m_train, 3)
factor = 5
X_train = np.random.rand(m_train, 3) * factor
Y_train = f(X_train)


B = m_train

# Construct the neural network
a0 = Data(ID = 'a0', shape = (B, 3))
x1 = Data(ID = 'x1', shape = (B, 3))

a1 = Data(ID = 'a1', shape = (B, 3))
rnn1_1 = RNN1(ID = 'RNN1_1', inputs = [a0, x1], output = a1)

y1 = Data(ID = 'y1', shape = (B, 1))
rnn1_2 = RNN2(ID = 'RNN1_2', inputs = [a1], output = y1)

# Attach loss layer
loss1 = Data(ID = 'loss1', shape = (1, 1))
sumloss1 = SumLoss(ID = 'Sumloss1', inputs = [y1], output = loss1)


# Add another RNN block
x2 = Data(ID = 'x2', shape = (B, 3))
a2 = Data(ID = 'a2', shape = (B, 3))
rnn2_1 = RNN1(ID = 'RNN2_1', inputs = [a1, x2], output = a2)

y2 = Data(ID = 'y2', shape = (B, 1))
rnn2_2 = RNN2(ID = 'RNN2_2', inputs = [a2], output = y2)

# Attach loss layer
loss2 = Data(ID = 'loss2', shape = (1, 1))
sumloss2 = SumLoss(ID = 'Sumloss2', inputs = [y2], output = loss2)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss1, loss2])


# # Visualize the network
# net.visualize(orientation = 'BT')


# Set required values
a0.val = a_train.copy()
x1.val = X_train.copy()
x2.val = X_train.copy()


# Share parameters
net.share_parameters([rnn1_1, rnn2_1])
net.share_parameters([rnn1_2, rnn2_2])

print(id(rnn1_1.Waa.val))
print(id(rnn2_1.Waa.val))
print(id(rnn1_1.Waa.deriv))
print(id(rnn2_1.Waa.deriv))



gradient_checker(net = net, data_obj = rnn1_1.Waa, h = 1e-6)


print(id(rnn1_1.Waa.val))
print(id(rnn2_1.Waa.val))
print(id(rnn1_1.Waa.deriv))
print(id(rnn2_1.Waa.deriv))