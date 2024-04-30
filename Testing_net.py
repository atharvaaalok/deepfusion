import numpy as np


from components.data.data import Data
from components.optimizers.sgd import SGD
from components.modules.matmul import MatMul
from components.modules.activation_functions.relu import Relu
from components.modules.loss_functions.mse import MSE
from components.net.net import Net


def f(X):
    Y = X[0, :] + 2 * X[1, :] ** 2 + 3 * X[2, :] ** 0.5
    return Y.reshape(1, -1)


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training and test data
m_train = 100000
m_test = 10

factor = 5
X_train = np.random.rand(3, m_train) * factor
Y_train = f(X_train)

X_test = np.random.rand(3, m_test) * factor
Y_test = f(X_test)


# Construct Neural Network
x = Data(ID = 'x', shape = (3, 1))
z1 = Data(ID = 'z1', shape = (10, 1))
matmul1 = MatMul(ID = 'matmul1', inputs = [x], output = z1)

a1 = Data(ID = 'a1', shape = (10, 1))
relu1 = Relu(ID = 'relu1', inputs = [z1], output = a1)

z2 = Data(ID = 'z2', shape = (10, 1))
matmul2 = MatMul(ID = 'matmul2', inputs = [a1], output = z2)

a2 = Data(ID = 'a2', shape = (10, 1))
relu2 = Relu(ID = 'relu2', inputs = [z2], output = a2)

z3 = Data(ID = 'z3', shape = (1, 1))
matmul3 = MatMul(ID = 'matmul3', inputs = [a2], output = z3)

y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
mse = MSE(ID = 'mse', inputs = [z3, y], output = loss)


regularizer_details = {'reg_strength': 0.0001, 'reg_name': 'L2'}

net = Net(ID = 'net', is_regularized = True, regularizer_details = regularizer_details)
net.add_nodes(matmul1, [x], z1)
net.add_nodes(relu1, [z1], a1)
net.add_nodes(matmul2, [a1], z2)
net.add_nodes(relu2, [z2], a2)
net.add_nodes(matmul3, [a2], z3)
net.add_nodes(mse, [z3, y], loss)

net.run_setup()

learning_rate = 1
net.set_learning_rate(learning_rate)


epochs = 1000
print_cost_every = 100

for epoch in range(epochs):
    x.val = X_train
    y.val = Y_train

    net.forward()

    if epoch % print_cost_every == 0:
        J = loss.val
        print(f'Epoch [{epoch}/{epochs}]. Cost: {J}')
    
    net.backward()

    net.update()