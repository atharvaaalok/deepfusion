import numpy as np

from ..components.data import Data
from ..components.modules import Flatten
from ..components.modules.loss_functions import SumLoss
from ..components.net import Net
from ..utils.grad_check import gradient_checker


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 3
factor = 5
a, b, c, d = 2, 3, 2, 3
X_train = np.random.rand(a, b, c, m_train) * factor


# Construct neural network
x = Data(ID = 'x', shape = (a, b, c))
inputs = [x]

z = Data(ID = 'z', shape = (a * b * c, 1))
flat = Flatten(ID = 'Flatten', inputs = inputs, output = z)

# Initialize, loss variable and attach loss function
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = SumLoss(ID = 'SumLoss', inputs = [z], output = loss)

# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set required values
x.val = X_train


gradient_checker(net = net, data_obj = x, loss_obj = loss, h = 1e-5)