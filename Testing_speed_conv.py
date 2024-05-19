import numpy as np

from .components.data import Data
from .components.net import Net
from .components.modules import Conv2D

from .components.modules.loss_functions import SumLoss
from .utils.colors import red, cyan, color_end
from .conv_data import *


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Get input and filter
# Inputs X1, X2 and filters F1, F2 are in file conv_data.py which have been imported above
n = 50
f = 3
padding = 0
stride = 1
filter_count = 2
X1 = np.random.rand(3, n, n)
X2 = np.random.rand(3, n, n)

B = 2

# Construct neural network
x = Data(ID = 'x', shape = (B, 3, n, n))
inputs = [x]

o = ((n + 2 * padding - f) // stride) + 1
z = Data(ID = f'z', shape = (B, filter_count, o, o))
conv = Conv2D(ID = f'Conv2D', inputs = inputs, output = z, filter_size = f, 
                filter_count = filter_count, padding = padding, stride = stride)

# Initialize, loss variable and attach loss function
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = SumLoss(ID = 'SumLoss', inputs = [z], output = loss)

# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Combine the inputs X1 and X2 into a single input with first dimension at the batch dimension
A = np.stack((X1, X2), axis = 0)

# Set the input and the filters for the net
x.val = A


# Set network training properties
epochs = 1
print_cost_every = 100
learning_rate = 0.01

net.set_learning_rate(learning_rate)


for epoch in range(1, epochs + 1):
    # Turn on training mode
    net.set_mode('train')

    # Generate mini batch of training examples
    # idx = np.random.choice(X_train.shape[0], size = B, replace = False)
    x.val = A
    # y.val = Y_train[idx, :]

    # Run the forward pass
    net.forward(verbose = True)            

    # Run the backward pass
    net.backward(verbose = True)

    # Update the parameters using the gradients
    net.update()

    # Print cost every few steps
    if epoch % print_cost_every == 0 or epoch == 1:
        J_train = loss.val
        num_digits = len(str(epochs))
        
        print(f'{red}Epoch:{color_end} [{epoch:{num_digits}}/{epochs}].  ' \
                f'{cyan}Train Cost:{color_end} {J_train:11.6f}.  ')