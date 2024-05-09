import numpy as np

from components.data import Data
from components.modules import MatMul
from components.modules.activation_functions import Relu
from components.modules.loss_functions import MSE
from components.net import Net

def f(X):
    Y = X[0, :] + 2 * X[1, :] ** 2 + 3 * X[2, :] ** 0.5
    return Y.reshape(1, -1)


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 100000
factor = 5
X_train = np.random.rand(3, m_train) * factor
Y_train = f(X_train)


# Construct neural network
x = Data(ID = 'x', shape = (3, 1))
inputs = [x]

layer_count = 3
for layer in range(1, layer_count + 1):
    z = Data(ID = f'z{layer}', shape = (10, 1))
    matmul = MatMul(ID = f'MatMul{layer}', inputs = inputs, output = z)
    
    a = Data(ID = f'a{layer}', shape = (10, 1))
    AF = Relu(ID = f'Relu{layer}', inputs = [z], output = a)

    inputs = [a]

# Attach final matrix multiplication layer
z = Data(ID = f'z{layer_count + 1}', shape = (1, 1))
matmul = MatMul(ID = f'MatMul{layer_count + 1}', inputs = inputs, output = z)

# Initialize target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
mse = MSE(ID = 'MSE', inputs = [z, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Train neural network
epochs = 1000
print_cost_every = 100
learning_rate = 0.01

net.set_learning_rate(learning_rate)

for epoch in range(epochs):
    x.val = X_train
    y.val = Y_train

    net.forward()

    if epoch % print_cost_every == 0:
        J = loss.val
        print(f'Epoch [{epoch}/{epochs}]. Cost: {J}')
    
    net.backward()

    net.update()