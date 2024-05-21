import numpy as np

# Import Net, Data and necessary modules
from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import Conv2D
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.pooling import MaxPool
from deepfusion.components.modules.flatten import Flatten
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.loss_functions import MSELoss
from deepfusion.utils.automate_training import automate_training


np.set_printoptions(linewidth = np.inf, precision = 2)
# Set seed for reproducibility
np.random.seed(0)


# Define a simple function for the CNN to learn
def f(X):
    return np.sum(X, axis = (1, 2, 3)).reshape(-1, 1)

## Generate training data
m_train = 100
c, h, w = 3, 32, 32
X_train = np.random.rand(m_train, c, h, w)
Y_train = f(X_train)


## Construct neural network
# Basic structure:
# x -> Conv2D -> z -> Relu -> a -> MaxPool -> a_pool -> Flatten -> a_flat -> MatMul -> h, + y -> MSE
# -> loss
x = Data(ID = 'x', shape = (1, c, h, w))

z = Data(ID = 'z', shape = (1, 5, 12, 12))
Conv = Conv2D(ID = 'Conv2D', inputs = [x], output = z,
              filter_size = 3, filter_count = 5, padding = 2, stride = 3)

a = Data(ID = 'a', shape = (1, 5, 12, 12))
ActF = Relu(ID = 'ActF', inputs = [z], output = a)

a_pool = Data(ID = 'a_pool', shape = (1, 5, 6, 6))
Pool = MaxPool(ID = 'Pool', inputs = [a], output = a_pool,
               filter_size = 2, padding = 0, stride = 2)

a_flat = Data(ID = 'a_flat', shape = (1, 5 * 6 * 6))
Flat = Flatten(ID = 'Flatten', inputs = [a_pool], output = a_flat)

h = Data(ID = 'h', shape = (1, 1))
Matmul = MatMul(ID = 'MatMul', inputs = [a_flat], output = h)

# Add target variable, loss variable and loss function
y = Data('y', shape = (1, 1))
loss = Data('loss', shape = (1, 1))
LossF = MSELoss(ID = 'LossF', inputs = [h, y], output = loss)

# Initialize the neural network
conv_net = Net(ID = 'Net', root_nodes = [loss])

## Visualize the network
conv_net.visualize()


# Profile the module times
x.val = X_train
y.val = Y_train
conv_net.forward(verbose = True)
conv_net.backward(verbose = True)
conv_net.clear_grads()


## Train the neural network automatically
automate_training(conv_net, X_train, Y_train, epochs = 1000, learning_rate = 0.1, lr_decay = 0)