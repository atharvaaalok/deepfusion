# In this demo we give a basic explanation of how to construct a vanilla recurrent neural network
# To convert a normal neural network into a RNN we need that the modules share parameters. This is
# achieved by calling the share_parameters() method of Net object and specifying the modules across
# which the parameters are to be shared.
# There are 2 modules that make up the RNN structure, RNN1 and RNN2. RNN1 calculates the next
# activation state a and RNN2 calculates the output predictions y using the next activation.

import numpy as np

from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules.rnn import RNN1, RNN2
from deepfusion.components.modules.loss_functions import SumLoss
from deepfusion.utils.grad_check import gradient_checker


np.set_printoptions(linewidth = np.inf)
# Set seed for reproducibility
np.random.seed(0)


## Generate training data
m_train = 1000
a_train = np.random.rand(m_train, 3)
X1_train = np.random.rand(m_train, 3)
X2_train = np.random.rand(m_train, 3)
# We will only explore gradient checking with RNN and use the SumLoss module for ease, therefore,
# there is no target variable Y needed


## Construct neural network
# Add first RNN block
a0 = Data(ID = 'a0', shape = (1, 3))
x1 = Data(ID = 'x1', shape = (1, 3))

a1 = Data(ID = 'a1', shape = (1, 3))
rnn1_1 = RNN1(ID = 'RNN1_1', inputs = [a0, x1], output = a1)

y1 = Data(ID = 'y1', shape = (1, 1))
rnn1_2 = RNN2(ID = 'RNN1_2', inputs = [a1], output = y1)

# Attach loss layer for first block
loss1 = Data(ID = 'loss1', shape = (1, 1))
sumloss1 = SumLoss(ID = 'Sumloss1', inputs = [y1], output = loss1)

# Add second RNN block
x2 = Data(ID = 'x2', shape = (1, 3))
a2 = Data(ID = 'a2', shape = (1, 3))
rnn2_1 = RNN1(ID = 'RNN2_1', inputs = [a1, x2], output = a2)

y2 = Data(ID = 'y2', shape = (1, 1))
rnn2_2 = RNN2(ID = 'RNN2_2', inputs = [a2], output = y2)

# Attach loss layer for second block
loss2 = Data(ID = 'loss2', shape = (1, 1))
sumloss2 = SumLoss(ID = 'Sumloss2', inputs = [y2], output = loss2)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss1, loss2])


## Visualize the network
net.visualize(orientation = 'BT') # Use BT = Bottom -> Top orientation for viewing RNN structure


# Set required values
a0.val = a_train
x1.val = X1_train
x2.val = X2_train


## Make the block share parameters
# This really is the step that makes the network a RNN
net.share_parameters([rnn1_1, rnn2_1])
net.share_parameters([rnn1_2, rnn2_2])


# Perform gradient checking
gradient_checker(net = net, data_obj = rnn1_1.Waa, h = 1e-6)