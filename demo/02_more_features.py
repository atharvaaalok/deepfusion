import numpy as np

# Import colors to decorate the terminal print statements
from deepfusion.utils.colors import red, cyan, color_end

# Import data and network definitions to keep this demo script simple and focus on functionalities
from demo_helper.data_def import generate_data
from demo_helper.net_simple import net_simple


np.set_printoptions(linewidth = np.inf, precision = 2)


# For this demo we use separate scripts to get our data and network so that we can focus on learning
# about the other features of the package in this script.

## Generate training data
m_train = 4
X_train, Y_train = generate_data(m_train)


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> MSE -> loss
net = net_simple


### Now we look at the different features available

## Visualize the network
net.visualize()


## Access the nodes - get node handle by passing in its unique ID
x = net.get_node('x')
y = net.get_node('y')
Matmul1 = net.get_node('Matmul1')
loss = net.get_node('loss')

# Print the nodes
print(x)
print(Matmul1.W)

# Accessing the values of the nodes
print('\nAccessing values of data objects\n' + 40 * '-')
print('loss.val before running forward pass:', loss.val)


# Run the forward pass
x.val = X_train
y.val = Y_train
net.forward()
print('loss.val after running forward pass:', loss.val, '\n')

# Run the backward pass
net.backward()
print('\nAccessing derivatives of data objects\n' + 40 * '-')
print('x.deriv:\n', x.deriv) # Every row here represents derivative w.r.t. that training example
print('Matmul1.W.deriv:\n', Matmul1.W.deriv, '\n')


## Using defined colors to print decorated output - use dummy values for epochs
epoch = 1
epochs = 100
J = loss.val
print('\nPrinting color decorated outputs\n' + 40 * '-')
print(f'{red}Epoch:{color_end} [{epoch}/{epochs}]. {cyan}Cost:{color_end} {J}.')


## Defining an optimizer
# Optimizer details are passed using a dictionary - specify 'optimizer_name' and 'hyperparameters'
optimizer_details = {'optimizer_name': 'RMSprop', 'hyperparameters': {'beta': 0.9}}
net.set_optimizer(optimizer_details)


## Defining a regularizer
# Regularizer details, like optimizer details are passed using a dictionary
regularizer_details = {'reg_name': 'L2', 'reg_strength': 0.001}
net.set_regularization(regularizer_details)


## Training the neural network
# With the optimizer and regularizer set we can now train the network as we did in the basic demo

# We can check the time taken by each module during the forward and backward pass by setting verbose
# = True
net.forward(verbose = True)
net.backward(verbose = True)