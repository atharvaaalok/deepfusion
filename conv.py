import numpy as np
from conv_data import *
from im2col_funcs import im2col, col2im

np.set_printoptions(linewidth = np.inf)
np.random.seed(0)


n = 7
filter_size = 3
padding = 0
stride = 1

filter_count = 2
o = (n + 2 * padding - filter_size) // stride + 1

minibatch = 2

F_shape = F1.shape
X_shape = X1.shape

# Attach filters into F_flat
F1_flat = F1.reshape(1, -1)
F2_flat = F2.reshape(1, -1)

F_flat = np.vstack((F1_flat, F2_flat))

# Attach inputs into X_flat
X1_flat = im2col(X1, filter_size, padding, stride)
X2_flat = im2col(X2, filter_size, padding, stride)

X_flat = np.stack((X1_flat, X2_flat), axis = 0)


# Find the convolution
conv_flat = np.einsum('il,jlk->jik', F_flat, X_flat)
conv_shape = conv_flat.shape
# print(conv_flat)

conv_shaped = conv_flat.reshape(-1, filter_count, o, o)
# print(conv_shaped)



## Backward Pass
conv_deriv_shaped = np.zeros(conv_shaped.shape)
conv_flat_deriv = conv_deriv_shaped.reshape(*conv_shape)


# Filter derivatives
F_flat_deriv = np.einsum('ebg,eag->ab', X_flat, conv_flat_deriv)
F_flat_deriv_shaped = np.vsplit(F_flat_deriv, filter_count)
F_i_flat_deriv = []
F_i_deriv = []
for i in range(filter_count):
    F_i_flat_deriv.append(F_flat_deriv_shaped[i])
    F_i_deriv.append(F_flat_deriv_shaped[i].reshape(F_shape))



# Input derivatives
X_flat_deriv = np.einsum('fb,afc->abc', F_flat, conv_flat_deriv)
X_i_flat_deriv = []
for i in range(minibatch):
    X_i_flat_deriv.append(X_flat_deriv[i])

X_i_deriv = []
for i in range(minibatch):
    deriv = col2im(X_i_flat_deriv[i], X_shape, filter_size, padding, stride)
    X_i_deriv.append(deriv)

print(X_i_deriv[0].shape)
print(X_shape)