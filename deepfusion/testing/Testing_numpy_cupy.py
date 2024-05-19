from ..utils.backend import Backend
Backend.set_backend('gpu')
np = Backend.get_array_module()


from ..components.data import Data
from ..components.modules import MatMul
from ..components.modules.activation_functions import Tanh
from ..components.modules.normalizations import BatchNorm
from ..components.modules.loss_functions import MSE
from ..components.net import Net
from ..utils.grad_check import gradient_checker


def f(X):
    Y = X[:, 0:1] + 2 * X[:, 1:2] ** 2 + 3 * X[:, 2:3] ** 0.5
    return Y


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Generate training data
m_train = 100000
factor = 5
X_train = np.random.rand(m_train, 3) * factor
Y_train = f(X_train)



# Construct neural network
ActF = Tanh
LossF = MSE
Norm = BatchNorm
weight_init_type = 'Random'

x = Data(ID = 'x', shape = (1, 3))
inputs = [x]

layer_count = 1
layer_size = 1000
for layer in range(1, layer_count + 1):
    z = Data(ID = f'z{layer}', shape = (1, layer_size))
    matmul = MatMul(ID = f'Matmul{layer}', inputs = inputs, output = z, weight_init_type = weight_init_type)

    a = Data(ID = f'a{layer}', shape = (1, layer_size))
    AF = ActF(ID = f'ActF{layer}', inputs = [z], output = a)

    a_norm = Data(ID = f'a_norm{layer}', shape = (1, layer_size))
    norm = Norm(ID = f'Norm{layer}', inputs = [a], output = a_norm)

    inputs = [a_norm]

# Attach final matrix multiplication layer
z = Data(ID = f'z{layer_count + 1}', shape = (1, 1))
matmul = MatMul(ID = f'MatMul{layer_count + 1}', inputs = inputs, output = z)

# Initialize target variable, loss variable and attach loss function
y = Data(ID = 'y', shape = (1, 1))
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = LossF(ID = 'LossF', inputs = [z, y], output = loss)


# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


# # Set required values
# x.val = X_train
# y.val = Y_train


# gradient_checker(net = net, data_obj = x, loss_obj = loss, h = 1e-6)


# Train neural network
epochs = 1000
print_cost_every = 100
learning_rate = 0.01

net.set_learning_rate(learning_rate)

for epoch in range(1, epochs + 1):
    # Fix a mini batch size
    B = 64
    selected_indices = np.random.choice(X_train.shape[0], size = B, replace = False)
    # Get a mini batch of the training data
    x.val = X_train[selected_indices, :]
    y.val = Y_train[selected_indices, :]

    net.forward(verbose = False)

    if epoch % print_cost_every == 0 or epoch == 1:
        J = loss.val
        print(f'Epoch [{epoch}/{epochs}]. Cost: {J}')
    
    net.backward(verbose = False)

    net.update()
    net.clear_grads()