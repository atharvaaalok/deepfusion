# Often when creating new modules, it is not possible or is very difficult to create and test its
# correctness by creating a neural network with the standard MSELoss or LogisticLoss etc. loss
# functions. Consider for e.g. that you are creating a Conv2D module, it's difficult to test this
# until you implement the flatten module so that a fully connected matrix multiplication layer can
# be attached to convert the output to a scalar value so that the standard MSELoss or LogisticLoss
# functions can be used. It would be really helpful if we could have a loss module that could
# convert any tensor to a scalar value without using specified target output labels. This is exactly
# what SumLoss loss function does. It simply takes in any dimensional tensor and adds all the values
# to produce a scalar output. Now we can backpropagate like usual and use gradient checking to test
# if the module that we just implemented (Conv2D for e.g.) has a correct implementation for the
# backward pass.

import numpy as np

from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import SumLoss
from deepfusion.utils.grad_check import gradient_checker


## Generate training data
m_train = 4
X_train = np.random.rand(m_train, 3)
# No need of target labels when using SumLoss loss function


## Construct neural network
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> SumLoss -> loss
x = Data(ID = 'x', shape = (1, 3))

z = Data(ID = 'z', shape = (1, 5))
Matmul = MatMul(ID = 'Matmul', inputs = [x], output = z)

a = Data(ID = 'a', shape = (1, 5))
ActF = Relu(ID = 'ActF', inputs = [z], output = a)

# Add loss variable and loss function
loss = Data('loss', shape = (1, 1))
LossF = SumLoss(ID = 'LossF', inputs = [a], output = loss)

# Initialize the neural network
net = Net(ID = 'Net', root_nodes = [loss])


# Set input node values
x.val = X_train


## Perform gradient checking
gradient_checker(net = net, data_obj = x, h = 1e-6)