import numpy as np

from .components.data import Data
from .components.modules.pooling.maxpool import MaxPool
from .components.modules.loss_functions import SumLoss
from .components.net import Net
from .utils.grad_check import gradient_checker


np.set_printoptions(linewidth = np.inf)

# Get input and filter
d = 3
n = 8
f = 2
padding = 0
stride = 2

A = np.random.randint(0, 1000, (d, n, n)) * 1.0
print(A)

# Construct neural network
x = Data(ID = 'x', shape = (3, n, n))
inputs = [x]

o = ((n + 2 * padding - f) // stride) + 1
z = Data(ID = 'z', shape = (d, o, o))
pool = MaxPool(ID = 'MaxPool', inputs = inputs, output = z, filter_size = f, padding = padding, stride = stride)

# Initialize, loss variable and attach loss function
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = SumLoss(ID = 'SumLoss', inputs = [z], output = loss)

# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set the required values
A = A[np.newaxis, ...]
x.val = A

net.forward()
print(z.val)

net.backward()
print(x.deriv)


gradient_checker(net = net, data_obj = x, loss_obj = loss, h = 1e-6)