# It is possible to have multiple loss functions in the network and to backpropagate through all of
# them. In this simple demo we attach two SumLoss modules at the last node and show the network
# construction and initialization procedure. In addition we also show how to work with multiple
# inputs to the network. We also see gradient checking in these scenarios.

import numpy as np

from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules import Add
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import SumLoss
from deepfusion.utils.grad_check import gradient_checker


# Set seed for reproducibility
np.random.seed(0)

## Generate training data
m_train = 4
X1_train = np.random.rand(m_train, 3)
X2_train = np.random.rand(m_train, 3)
# No need of target labels when using SumLoss loss function


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> SumLoss -> loss
x1 = Data(ID = 'x1', shape = (1, 3))
x2 = Data(ID = 'x2', shape = (1, 3))

x1_plus_x2 = Data(ID = 'x1_plus_x2', shape = (1, 3))
add_module = Add(ID = 'Add', inputs = [x1, x2], output = x1_plus_x2)

z = Data(ID = 'z', shape = (1, 5))
Matmul = MatMul(ID = 'Matmul', inputs = [x1_plus_x2], output = z)

a = Data(ID = 'a', shape = (1, 5))
ActF = Relu(ID = 'ActF', inputs = [z], output = a)

# Add loss variable and loss function
loss1 = Data('loss1', shape = (1, 1))
LossF1 = SumLoss(ID = 'LossF1', inputs = [a], output = loss1)

loss2 = Data('loss2', shape = (1, 1))
LossF2 = SumLoss(ID = 'LossF2', inputs = [a], output = loss2)

# Initialize the neural network
# Specify all the loss nodes when initializing the network
net = Net(ID = 'Net', root_nodes = [loss1, loss2])
# Run this code again by specifying only loss1 or loss 2 and check that the gradients half


## Visualize the network
net.visualize('multiple_inputs_multiple_loss')


# Set input node values
x1.val = X1_train
x2.val = X2_train


## Perform gradient checking
# Gradient checking with multiple loss functions adds the gradients due to the different losses.
gradient_checker(net = net, data_obj = x1, h = 1e-6)