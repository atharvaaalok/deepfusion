import numpy as np


from components.data.data import Data
from components.optimizers.sgd import SGD
from components.modules.matmul import MatMul

from components.modules.activation_functions.relu import Relu
from components.modules.activation_functions.sigmoid import Sigmoid
from components.modules.activation_functions.tanh import Tanh
from components.modules.activation_functions.lrelu import LRelu
from components.modules.activation_functions.prelu import PRelu
from components.modules.activation_functions.elu import ELU

from components.modules.loss_functions.mse import MSE
from components.net.net_new import Net


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
weight_init_type = 'Random'
ActF = Tanh

x = Data(ID = 'x', shape = (3, 1))
z1 = Data(ID = 'z1', shape = (10, 1))
matmul1 = MatMul(ID = 'matmul1', inputs = [x], output = z1, weight_init_type = weight_init_type)

a1 = Data(ID = 'a1', shape = (10, 1))
AF1 = ActF(ID = 'AF1', inputs = [z1], output = a1)

z2 = Data(ID = 'z2', shape = (10, 1))
matmul2 = MatMul(ID = 'matmul2', inputs = [a1], output = z2, weight_init_type = weight_init_type)

a2 = Data(ID = 'a2', shape = (10, 1))
AF2 = ActF(ID = 'AF2', inputs = [z2], output = a2)

z3 = Data(ID = 'z3', shape = (1, 1))
matmul3 = MatMul(ID = 'matmul3', inputs = [a2], output = z3, weight_init_type = weight_init_type)

y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
mse = MSE(ID = 'mse', inputs = [z3, y], output = loss)


regularizer_details = {'reg_strength': 0.0001, 'reg_name': 'L2'}
optimizer_details = {'optimizer_name': 'Adam', 'hyperparameters': {}}


net = Net(ID = 'net', root_nodes = [loss], optimizer_details = optimizer_details, is_regularized = True, regularizer_details = regularizer_details)

print([ele.ID for ele in net.topological_order])
print()
print(net.graph_visual)
print()


learning_rate = 0.01
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