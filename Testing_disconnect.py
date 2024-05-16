from .components.data import Data
from .components.modules import MatMul
from .components.modules.activation_functions import Tanh
from .components.modules.loss_functions import MSE
from .components.net import Net


# Construct neural network
ActF = Tanh
LossF = MSE
weight_init_type = 'Random'

x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

z1 = Data(ID = f'z1', shape = (1, 10))
matmul1 = MatMul(ID = f'Matmul1', inputs = inputs, output = z1, weight_init_type = weight_init_type)

a1 = Data(ID = f'a1', shape = (1, 10))
act_f1 = ActF(ID = f'ActF1', inputs = [z1], output = a1)


z2 = Data(ID = f'z2', shape = (1, 1))
matmul2 = MatMul(ID = f'MatMul2', inputs = [a1], output = z2)

# Add target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = LossF(ID = 'LossF', inputs = [z2, y], output = loss)

# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Visualize initial network
net.visualize('Initial')


# Remove loss connection and visualize
net.disconnect(data_obj = z2, module_obj = sum_loss)
net.visualize('No Loss')