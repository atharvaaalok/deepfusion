# Import Net, Data and necessary modules
from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import MSELoss


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> MSE -> loss
x = Data(ID = 'x', shape = (1, 3))

z1 = Data(ID = 'z1', shape = (1, 5))
Matmul1 = MatMul(ID = 'Matmul1', inputs = [x], output = z1)

a = Data(ID = 'a', shape = (1, 5))
ActF = Relu(ID = 'ActF', inputs = [z1], output = a)

z2 = Data(ID = 'z2', shape = (1, 1))
Matmul2 = MatMul(ID = 'Matmul2', inputs = [a], output = z2)

# Add target variable, loss variable and loss function
y = Data('y', shape = (1, 1))
loss = Data('loss', shape = (1, 1))
LossF = MSELoss(ID = 'LossF', inputs = [z2, y], output = loss)

# Initialize the neural network
net_simple = Net(ID = 'Net', root_nodes = [loss])