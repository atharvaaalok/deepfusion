# Save a model after some training, load it and train forward from pre-trained state

import numpy as np

# Import automatic training utility
from deepfusion.utils.automate_training import automate_training
from deepfusion.components.net import Net

# Import data and network definitions to keep this demo script simple and focus on functionalities
from demo_helper.data_def import generate_data
from demo_helper.net_simple import net_simple


## Generate training and validation data
m_train = 64
X_train, Y_train = generate_data(m_train)


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> MSE -> loss
net = net_simple

## Train the neural network
# Train for a 1000 epochs and save the model
automate_training(net, X_train, Y_train, epochs = 1000)


## Save the model - the extension is .df
net.save(file_path = 'partially_trained_model.df')
print('Saving the model after training for 1000 epochs.\n')

## Load the saved model and train forward - use the Net class to load
my_net = Net.load(file_path = 'partially_trained_model.df')


## Train the neural network
print('Loading a saved model and training for 1000 epochs.')
automate_training(my_net, X_train, Y_train, epochs = 1000)