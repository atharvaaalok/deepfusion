import numpy as np

from .components.data import Data
from .components.modules.conv2d_new import Conv2D
from .components.modules.loss_functions import SumLoss
from .components.net import Net
from .utils.grad_check import gradient_checker


def conv2D(A, F, padding, stride):
    
    n, _ = A.shape
    f, _ = F.shape

    # Pad A
    A = np.pad(A, padding)

    o = (n + 2 * padding - f) // stride + 1
    B = np.zeros((o, o))

    for i in range(o):
        for j in range(o):
            idx_i, idx_j = i * stride, j * stride
            mat = A[idx_i: idx_i + f, idx_j: idx_j + f]
            B[i, j] = np.sum(mat * F)
    
    return B


np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


# Get input and filter
A = np.array([[2, 3, 7, 4, 6, 2, 9],
              [6, 6, 9, 8, 7, 4, 3],
              [3, 4, 8, 3, 8, 9, 7],
              [7, 8, 3, 6, 6, 3, 4],
              [4, 2, 1, 8, 3, 4, 6],
              [3, 2, 4, 1, 9, 8, 3],
              [0, 1, 3, 9, 2, 1, 4]], dtype = np.float64)

F = np.array([[3, 4, 4],
              [1, 0, 2],
              [-1, 0, 3]], dtype = np.float64)


n = 7
f = 3
padding = 0
stride = 1

B = conv2D(A, F, padding, stride)
print(B)


# Construct neural network
x = Data(ID = 'x', shape = (1, n, n))
inputs = [x]

o = ((n + 2 * padding - f) // stride) + 1
z = Data(ID = f'z', shape = (1, o, o))
conv = Conv2D(ID = f'Conv2D', inputs = inputs, output = z, filter_size = f, 
                filter_count = 1, padding = padding, stride = stride)

# Initialize, loss variable and attach loss function
loss = Data(ID = 'loss', shape = (1, 1))
sum_loss = SumLoss(ID = 'SumLoss', inputs = [z], output = loss)

# Initialize neural network
net = Net(ID = 'Net', root_nodes = [loss])


A = A[np.newaxis, ...]
A = A[np.newaxis, ...]
F = F[np.newaxis, ...]


x.val = A
conv.parameter_list[0].val = F

net.forward()
print(z.val)

net.backward()
print(x.deriv.shape)

gradient_checker(net = net, data_obj = x, loss_obj = loss, h = 1e-6)