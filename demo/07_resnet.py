# Constructing and training a residual network

import numpy as np

# Import Net, Data and necessary modules
from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import MSELoss
from deepfusion.components.modules import Add
from deepfusion.utils.automate_training import automate_training

# Import data definitions to keep this demo script simple and focus on network construction
from demo_helper.data_def import generate_data


## Generate training and validation data
m_train = 100000
m_validation = 100
X_train, Y_train = generate_data(m_train)
X_val, Y_val = generate_data(m_validation)


## Construct neural network
x = Data(ID = 'x', shape = (1, 3))
xin = x

layer_count = 3
# Define a residual block and repeat through for loop
for layer in range(1, layer_count + 1):
    z1 = Data(ID = f'z1_{layer}', shape = (1, 5))
    MatrixMul = MatMul(ID = f'Matmul1_{layer}', inputs = [xin], output = z1)

    a1 = Data(ID = f'a1_{layer}', shape = (1, 5))
    ActF = Relu(ID = f'ActF1_{layer}', inputs = [z1], output = a1)

    z2 = Data(ID = f'z2_{layer}', shape = (1, 5))
    MatrixMul = MatMul(ID = f'Matmul2_{layer}', inputs = [a1], output = z2)

    if layer == 1:
        # Don't use residual block for the first layer, as there is a size incompatibility (3 and 5)
        a2 = Data(ID = f'a2_{layer}', shape = (1, 5))
        ActF = Relu(ID = f'ActF2_{layer}', inputs = [z2], output = a2)

        xin = a2
        continue
    
    # For any other layer except layer 1, create the residual connection
    z2_plus_xin = Data(ID = f'{z2.ID}_{xin.ID}_{layer}', shape = (1, 5))
    add = Add(ID = f'Add_{layer}', inputs = [z2, xin], output = z2_plus_xin)
    
    a2 = Data(ID = f'a2_{layer}', shape = (1, 5))
    ActF = Relu(ID = f'ActF2_{layer}', inputs = [z2_plus_xin], output = a2)

    xin = a2

# Attach matrix multiplication layer to bring activations to scalar values
z = Data(ID = f'z_{layer_count + 1}', shape = (1, 1))
MatrixMul = MatMul(ID = f'Matmul_{layer_count + 1}', inputs = [xin], output = z)

# Add target variable, loss variable and loss function
y = Data('y', shape = (1, 1))
loss = Data('loss', shape = (1, 1))
LossF = MSELoss(ID = 'LossF', inputs = [z, y], output = loss)

# Initialize the neural network
resnet = Net(ID = 'Net', root_nodes = [loss])


## Visualize neural network
resnet.visualize()


# Train the neural network automatically
automate_training(resnet, X_train, Y_train, epochs = 1000)