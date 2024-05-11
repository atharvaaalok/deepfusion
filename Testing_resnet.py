import numpy as np

from components.data import Data
from components.modules import MatMul
from components.modules import Add
from components.modules.activation_functions import Relu
from components.modules.loss_functions import MSE
from components.net import Net


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Construct neural network
x = Data(ID = 'x', shape = (3, 1))
xin = x

layer_count = 2
for layer in range(1, layer_count + 1):
    z1 = Data(ID = f'z1_{layer}', shape = (10, 1))
    matmul = MatMul(ID = f'MatMul1_{layer}', inputs = [xin], output = z1)
    
    a1 = Data(ID = f'a1_{layer}', shape = (10, 1))
    AF = Relu(ID = f'Relu1_{layer}', inputs = [z1], output = a1)

    z2 = Data(ID = f'z2_{layer}', shape = (10, 1))
    matmul = MatMul(ID = f'MatMul2_{layer}', inputs = [a1], output = z2)

    z2_plus_xin = Data(ID = f'{z2.ID}_{xin.ID}_{layer}', shape = (10, 1))
    add = Add(ID = f'Add{layer}', inputs = [z2, xin], output = z2_plus_xin)

    a2 = Data(ID = f'a2_{layer}', shape = (10, 1))
    AF = Relu(ID = f'Relu2_{layer}', inputs = [z2_plus_xin], output = a2)

    xin = a2

# Attach final matrix multiplication layer
z = Data(ID = f'z{layer_count + 1}', shape = (1, 1))
matmul = MatMul(ID = f'MatMul{layer_count + 1}', inputs = [xin], output = z)

# Initialize target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
mse = MSE(ID = 'MSE', inputs = [z, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Visualize the neural network
net.visualize()