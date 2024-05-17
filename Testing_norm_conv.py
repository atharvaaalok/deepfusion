import numpy as np

from .components.data import Data
from .components.modules import Conv2D
from .components.modules.activation_functions import Relu
from .components.modules.normalizations import BatchNorm
from .components.modules.loss_functions import SumLoss
from .components.net import Net
from .utils.grad_check import gradient_checker
from .conv_data import *


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Get input and filter
# Inputs X1, X2 and filters F1, F2 are in file conv_data.py which have been imported above
n = 7
f = 3
padding = 0
stride = 1
filter_count = 2

weight_init_type = 'He'

# Construct neural network
x = Data(ID = 'x', shape = (3, n, n))
inputs = [x]

o = ((n + 2 * padding - f) // stride) + 1
z = Data(ID = f'z', shape = (filter_count, o, o))
conv = Conv2D(ID = f'Conv2D', inputs = inputs, output = z, filter_size = f, 
                filter_count = filter_count, padding = padding, stride = stride, weight_init_type = weight_init_type)

z_norm = Data(ID = f'z_norm', shape = (filter_count, o, o))
norm = BatchNorm(ID = f'Norm', inputs = [z], output = z_norm)

a = Data(ID = 'a', shape = (filter_count, o, o))
act_f = Relu(ID = 'Tanh', inputs = [z_norm], output = a)

# Initialize, loss variable and attach loss function
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = SumLoss(ID = 'SumLoss', inputs = [a], output = loss)

# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Combine the inputs X1 and X2 into a single input with first dimension at the batch dimension
A = np.stack((X1, X2), axis = 0)

# Set the input and the filters for the net
x.val = A
conv.parameter_list[0].val = F1
conv.parameter_list[1].val = F2


net.forward()
print(z.val)

net.backward()
print(x.deriv.shape)


gradient_checker(net = net, data_obj = x, loss_obj = loss, h = 1e-6)