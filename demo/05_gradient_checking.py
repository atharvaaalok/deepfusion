# DeepFusion provides a gradient checking utility using which you can compare the analytic and
# numerical gradient of the loss w.r.t. any data object in the network and verify correctness of the
# backward pass

import numpy as np

# Import gradient checking utility
from deepfusion.utils.grad_check import gradient_checker

# Import data and network definitions to keep this demo script simple and focus on functionalities
from demo_helper.data_def import generate_data
from demo_helper.net_simple import net_simple

np.set_printoptions(precision = 2)

## Generate training and validation data
m_train = 3
X_train, Y_train = generate_data(m_train)


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> MSE -> loss
net = net_simple

# Get the nodes w.r.t. which we want to perform gradient checking
x = net.get_node('x')
y = net.get_node('y')
Matmul1 = net.get_node('Matmul1')


# Set input node values
x.val = X_train
y.val = Y_train

## Perform gradient checking
gradient_checker(net, data_obj = x, h = 1e-6)

gradient_checker(net, data_obj = Matmul1.W, h = 1e-6)