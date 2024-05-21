# By default the deepfusion package works with numpy. But everything can be run on the GPU by simply
# specifying the backend to be cupy instead of numpy. Instead of 'import numpy as np' do as follows:

from deepfusion.utils.backend import Backend
Backend.set_backend('gpu')      # Set the backend to 'gpu', by default it is 'cpu'
np = Backend.get_array_module() # np is numpy if backend is 'cpu', else cupy if backend is 'gpu'
# Note that this works as numpy and cupy have similar API

# Import automatic training utility
from deepfusion.utils.automate_training import automate_training

# Import data and network definitions to keep this demo script simple and focus on functionalities
from demo_helper.data_def import generate_data
from demo_helper.net_simple import net_simple

np.random.seed(0)
## Generate training and validation data
m_train = 100000
X_train, Y_train = generate_data(m_train)

## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> MSE -> loss
net = net_simple


## Train the neural network

# Train the neural network automatically
automate_training(net, X_train, Y_train, epochs = 1000)