# Now we add more layers and more modules to our simple net

import numpy as np

# Import Net, Data and necessary modules
from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import MSELoss
from deepfusion.components.modules.normalizations import BatchNorm
from deepfusion.utils.printing import print_net_performance

# Import data definitions to keep this demo script simple and focus on network construction
from demo_helper.data_def import generate_data


## Generate training and validation data
m_train = 100000
m_validation = 100
X_train, Y_train = generate_data(m_train)
X_val, Y_val = generate_data(m_validation)


## Construct neural network
x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

layer_count = 3
layer_size = 5
for layer in range(1, layer_count + 1):
    z = Data(ID = f'z_{layer}', shape = (1, layer_size))
    MatrixMul = MatMul(ID = f'Matmul_{layer}', inputs = inputs, output = z)

    a = Data(ID = f'a_{layer}', shape = (1, layer_size))
    ActF = Relu(ID = f'ActF_{layer}', inputs = [z], output = a)

    a_norm = Data(ID = f'a_norm_{layer}', shape = (1, layer_size))
    Norm = BatchNorm(ID = f'BatchNorm_{layer}', inputs = [a], output = a_norm)

    inputs = [a_norm]

# Attach matrix multiplication layer to bring activations to scalar values
z = Data(ID = f'z_{layer_count + 1}', shape = (1, 1))
MatrixMul = MatMul(ID = f'Matmul_{layer_count + 1}', inputs = inputs, output = z)

# Add target variable, loss variable and loss function
y = Data('y', shape = (1, 1))
loss = Data('loss', shape = (1, 1))
LossF = MSELoss(ID = 'LossF', inputs = [z, y], output = loss)

# Initialize the neural network
net = Net(ID = 'Net', root_nodes = [loss])


## Visualize neural network
net.visualize()


## Train the neural network
# Note that batchnorm works differently and train and test and therefore the 'mode' of the network
# is to be set appropriately when dealing with training data or evaluating on validation data

# Set training properties
epochs = 1000
print_cost_every = 100
learning_rate = 0.001
B = 64

net.set_learning_rate(learning_rate)

for epoch in range(1, epochs + 1):
    # Turn on training mode
    net.set_mode('train')

    # Get a mini-batch of the data
    idx = np.random.choice(X_train.shape[0], size = B, replace = False)
    # Feed in the data to the input nodes of the network for forward pass to be performed
    x.val = X_train[idx, :]
    y.val = Y_train[idx, :]

    # Run forward pass
    net.forward()

    # Run backward pass
    net.backward()

    # Update the network parameters
    net.update()

    # Print the cost during training every few iterations
    if epoch % print_cost_every == 0 or epoch == 1:
        J_train = loss.val

        # Turn on testing mode and evaluate model on validation data
        net.set_mode('test')
        x.val = X_val
        y.val = Y_val
        net.forward()
        J_val = loss.val

        # The package also provides a function for decorated printing
        print_net_performance(epochs = epochs, epoch = epoch, J_train = J_train, J_val = J_val)