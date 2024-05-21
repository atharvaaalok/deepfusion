import numpy as np

# Import Net, Data and necessary modules
from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import MSELoss


np.set_printoptions(linewidth = np.inf)
# Set seed for reproducibility
np.random.seed(0)


## Generate training data
# We will use the following function as a running example in the demo scripts
def f(X):
    # For a training example - y = x1 + 2 * x2^2 + 3 * x3^.5, the fancy indexing preserves 2D shape
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y

# We will use m for number of examples
m_train = 100000
factor = 5
X_train = np.random.rand(m_train, 3) * factor
Y_train = f(X_train)


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
net = Net(ID = 'Net', root_nodes = [loss])


# To keep this script simple we don't define an optimizer, we see this in the next demo
# By default the Adam optimizer will be used.


## Train the neural network
# Set training properties
epochs = 1000
print_cost_every = 100
learning_rate = 0.001
B = 64

net.set_learning_rate(learning_rate)

for epoch in range(1, epochs + 1):
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
        J = loss.val
        print(f'Epoch: [{epoch}/{epochs}]. Cost: {J}.')