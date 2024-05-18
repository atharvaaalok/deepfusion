import numpy as np

from .components.data import Data
from .components.net import Net
from .components.modules.activation_functions import Tanh
from .components.modules.loss_functions import SumLoss
from .utils.colors import red, cyan, color_end


def f(X):
    Y = np.sum(X ** 2 + 3 * X ** 3 + X, axis = 1, keepdims = True)
    return Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 10000000
factor = 5
dim = 3
X_train = np.random.rand(m_train, dim) * factor
Y_train = f(X_train)


B = m_train

# Construct neural network
ActF = Tanh
LossF = SumLoss

x = Data(ID = 'x', shape = (B, dim))
inputs = [x]

a = Data(ID = 'a', shape = (B, dim))
act_f = ActF(ID = 'ActF', inputs = inputs, output = a)

# Attach loss variable and module
loss = Data(ID = 'loss', shape = (1, 1))
loss_f = LossF(ID = 'LossF', inputs = [a], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set network training properties
epochs = 1
print_cost_every = 100
learning_rate = 0.01

net.set_learning_rate(learning_rate)


for epoch in range(1, epochs + 1):
    # Turn on training mode
    net.set_mode('train')

    # Generate mini batch of training examples
    idx = np.random.choice(X_train.shape[0], size = B, replace = False)
    x.val = X_train[idx, :]
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