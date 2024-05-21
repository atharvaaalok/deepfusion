# The only change when doing logistic regression when compared to linear regression is the different
# loss function, LogisticLoss (and the different dataset of course).

import numpy as np

# Import Net, Data and necessary modules
from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import LogisticLoss
from deepfusion.utils.automate_training import automate_training

# Import data and network definitions to keep this demo script simple and focus on functionalities
from demo_helper.data_def import generate_moons_data, plot_moons_data
from demo_helper.net_simple import net_simple


np.set_printoptions(linewidth = np.inf)
# Set seed for reproducibility
np.random.seed(0)


## Generate training data
m_train = 10000
num_classes = 2
X_train, Y_train = generate_moons_data(num_samples = m_train, num_classes = num_classes)
plot_moons_data(X_train, Y_train)


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> LogisticLoss -> loss
x = Data(ID = 'x', shape = (1, 2))

z1 = Data(ID = 'z1', shape = (1, 5))
Matmul1 = MatMul(ID = 'Matmul1', inputs = [x], output = z1)

a = Data(ID = 'a', shape = (1, 5))
ActF = Relu(ID = 'ActF', inputs = [z1], output = a)

z2 = Data(ID = 'z2', shape = (1, 1))
Matmul2 = MatMul(ID = 'Matmul2', inputs = [a], output = z2)

# Add target variable, loss variable and loss function
y = Data('y', shape = (1, 1))
loss = Data('loss', shape = (1, 1))
LossF = LogisticLoss(ID = 'LossF', inputs = [z2, y], output = loss)

# Initialize the neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Train the neural network automatically
automate_training(net, X_train, Y_train, epochs = 1000)