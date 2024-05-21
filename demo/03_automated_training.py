# The package provides a utility to automate neural network training and plots progress curves

import numpy as np

# Import automatic training utility
from deepfusion.utils.automate_training import automate_training

# Import data and network definitions to keep this demo script simple and focus on functionalities
from demo_helper.data_def import generate_data
from demo_helper.net_simple import net_simple


## Generate training and validation data
m_train = 100000
m_validation = 100
X_train, Y_train = generate_data(m_train)
X_val, Y_val = generate_data(m_validation)


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> MSE -> loss
net = net_simple


## Train the neural network
# Set training properties
epochs = 1000
print_cost_every = 100
learning_rate = 0.001
B = 64
learning_rate_decay = 0.01

# Train the neural network automatically
automate_training(net, X_train, Y_train,
                  X_val = X_val, Y_val = Y_val,
                  B = B,
                  epochs = epochs,
                  print_cost_every = print_cost_every,
                  learning_rate = learning_rate,
                  lr_decay = learning_rate_decay)