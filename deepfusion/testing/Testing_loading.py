import numpy as np

from ..components.net import Net
from .data_def import generate_data


# Load the pre-trained model
file_path = 'deepfusion/First_model'
net = Net.load(file_path)


# Train the model further

np.set_printoptions(linewidth = np.inf)
np.random.seed(0)

# Get training, validation and test data
m_train = 100000
m_val = 10
m_test = 10
X_train, Y_train = generate_data(m_train)
X_val, Y_val = generate_data(m_val)
X_test, Y_test = generate_data(m_test)


# Set network training properties
epochs = 500
print_cost_every = 100
learning_rate = 0.01
B = 64
learning_rate_decay = 0.001


# Train the neural network automatically
net.automate_training(X_train, Y_train, X_val, Y_val,
                      B = B,
                      epochs = epochs,
                      print_cost_every = print_cost_every,
                      learning_rate = learning_rate,
                      lr_decay = learning_rate_decay)