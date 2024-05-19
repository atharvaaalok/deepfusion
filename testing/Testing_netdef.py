import numpy as np

from ..components.data import Data
from ..components.net import Net
from ..components.modules import MatMul
from ..components.modules.activation_functions import Relu
from ..components.modules.loss_functions import MSE
from ..utils.colors import red, cyan, color_end


def f(X):
    Y = np.sum(X ** 2 + 3 * X ** 3 + X, axis = 1, keepdims = True)
    return Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 100000
factor = 5
dim = 3
X_train = np.random.rand(m_train, dim) * factor
Y_train = f(X_train)


# Construct neural network
ActF = Relu
LossF = MSE

x = Data(ID = 'x', shape = (1, dim))
inputs = [x]

z1 = Data(ID = 'z1', shape = (1, dim))
matmul1 = MatMul(ID = 'MatMul1', inputs = inputs, output = z1)

a = Data(ID = 'a', shape = (1, dim))
act_f = ActF(ID = 'ActF', inputs = [z1], output = a)

z2 = Data(ID = 'z2', shape = (1, 1))
matmul2 = MatMul(ID = 'MatMul2', inputs = [a], output = z2)

# Attach loss variable and module
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
loss_f = LossF(ID = 'LossF', inputs = [z2, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set network training properties
epochs = 1000
print_cost_every = 100
learning_rate = 0.01
B = 64

net.set_learning_rate(learning_rate)

# Turn on training mode
net.set_mode('train')

for epoch in range(1, epochs + 1):

    if epoch < 500:
        B = 64
    else:
        B = m_train

    # Generate mini batch of training examples
    idx = np.random.choice(X_train.shape[0], size = B, replace = False)
    x.val = X_train[idx, :]
    y.val = Y_train[idx, :]

    # Run the forward pass
    net.forward()            

    # Run the backward pass
    net.backward()

    # Update the parameters using the gradients
    net.update()

    # Print cost every few steps
    if epoch % print_cost_every == 0 or epoch == 1:
        J_train = loss.val
        num_digits = len(str(epochs))
        
        print(f'{red}Epoch:{color_end} [{epoch:{num_digits}}/{epochs}].  ' \
                f'{cyan}Train Cost:{color_end} {J_train:11.6f}.  ')