from ..components.data import Data
from ..components.modules import MatMul
from ..components.modules.activation_functions import Tanh
from ..components.modules.loss_functions import MSE
from ..components.net import Net


# Construct neural network
ActF = Tanh
LossF = MSE
weight_init_type = 'Random'

x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

layer_count = 2
layer_size = 10
for layer in range(1, layer_count + 1):
    z = Data(ID = f'z{layer}', shape = (1, layer_size))
    matmul = MatMul(ID = f'Matmul{layer}', inputs = inputs, output = z, weight_init_type = weight_init_type)

    a = Data(ID = f'a{layer}', shape = (1, layer_size))
    act_f = ActF(ID = f'ActF{layer}', inputs = [z], output = a)

    inputs = [a]

# Attach final matrix multiplication layer
z = Data(ID = f'z{layer_count + 1}', shape = (1, 1))
matmul = MatMul(ID = f'MatMul{layer_count + 1}', inputs = inputs, output = z)

# Initialize target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = LossF(ID = 'LossF', inputs = [z, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])