import numpy as np

from .components.data import Data
from .components.modules import MatMul
from .components.modules.activation_functions import Tanh
from .components.modules.normalizations import BatchNorm
from .components.modules.dropout import InvertedDropout
from .components.modules.loss_functions import Logistic
from .components.net import Net
from .utils.grad_check import gradient_checker


def generate_moons(n_samples, noise=0.1):
    # Generate random angles
    random_angles = np.random.rand(n_samples) * np.pi

    # Outer circle
    outer_radius = 10
    outer_x = outer_radius * np.cos(random_angles)
    outer_y = outer_radius * np.sin(random_angles)

    # Inner circle (turned upside down)
    inner_radius = 5
    inner_x = inner_radius * np.cos(random_angles) + outer_radius
    inner_y = -inner_radius * np.sin(random_angles)  # Turning upside down

    # Add noise
    outer_x += np.random.normal(scale=noise, size=n_samples)
    outer_y += np.random.normal(scale=noise, size=n_samples)
    inner_x += np.random.normal(scale=noise, size=n_samples)
    inner_y += np.random.normal(scale=noise, size=n_samples)

    # Combine inner and outer circles
    X = np.vstack([np.hstack([outer_x, inner_x]), np.hstack([outer_y, inner_y])])
    Y = np.hstack([np.zeros(n_samples), np.ones(n_samples)]).astype(int).reshape(1, -1)

    return X, Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate moons dataset
m_train = 1000
X_train, Y_train = generate_moons(n_samples = m_train)


# Construct neural network
x = Data(ID = 'x', shape = (2, 1))
inputs = [x]

z = Data(ID = f'z', shape = (10, 1))
matmul = MatMul(ID = f'MatMul', inputs = inputs, output = z, weight_init_type = 'Random')

a = Data(ID = f'a', shape = (10, 1))
act_f = Tanh(ID = f'ActF', inputs = [z], output = a)

z2 = Data(ID = f'z2', shape = (1, 1))
mat = MatMul(ID = f'MatMul2', inputs = [a], output = z2, weight_init_type = 'Random')

# Initialize target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
mse = Logistic(ID = 'MSE', inputs = [z2, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set required values
x.val = X_train
y.val = Y_train


gradient_checker(net = net, data_obj = z2, loss_obj = loss, h = 1e-5)
print()

# Train neural network
epochs = 1000
print_cost_every = 100
learning_rate = 0.001

net.set_learning_rate(learning_rate)
net.set_mode('train')

for epoch in range(1, epochs + 1):
    x.val = X_train
    y.val = Y_train

    net.forward()

    if epoch % print_cost_every == 0 or epoch == 1:
        J = loss.val
        print(f'Epoch [{epoch}/{epochs}]. Cost: {J}')
    
    net.backward()

    net.update()