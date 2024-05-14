import numpy as np
from typing import override

from .module import Module
from ..data.data import Data
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS


class Conv2D(Module):
    """Convolution 2D module that given an input X and a defined filter size and filter count
    produces an activation map via convolving the filters on the input.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the modules output after transforming the input values.
        parameter_list:
            List of parameter Data objects associated with the module. Empty if module has no
            parameters.
        learning_rate:
            Controls the step size of the update for the module's parameters.
        is_frozen:
            Boolean that decides if updates will be made to module's parameters or not.
        optimizer_details:
            Dictionary containing name of the optimizer and a dictionary of its associated
            hyperparameters.
        filter_size:
            Height and width of the filter that is convolved over the input.
        filter_count:
            Number of filters that are convolved over the input to produce an activation map.
        padding:
            Amount of 0 padding to the edges of the input.
        stride:
            Step size of the filter over the input.
    """

    different_at_train_test = False
    is_regularizable = True


    def __init__(
        self,
        ID: str,
        inputs: list[Data],
        output: Data,
        filter_size: int,
        filter_count: int,
        padding: int = 0,
        stride: int = 1,
        learning_rate: float = 1e-6,
        is_frozen: bool = False,
        optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS,
    ) -> None:
        """Initializes Conv2D module based on ID, inputs, output, filter size, filter count and
        other optional parameters."""

        # Get input dimensions
        D_in, H_in, W_in = inputs[0].shape
        D_out, H_out, W_out = output.shape

        # Go through checks first
        assert D_out == filter_count, 'Conv2D output depth should be same as filter count.'
        assert H_in == W_in, 'Conv2D input should be square.'
        assert H_out == W_out, 'Conv2D output should be square.'
        assert (H_in + 2 * padding - filter_size) % stride == 0, \
            'Conv2D input size, filter size, padding and stride must satisfy (n + 2p - f) % s = 0.'
        assert ((H_in + 2 * padding - filter_size) / stride) + 1 == H_out, \
            'Conv2D output size should be consistent with ((n_in + 2p - f) / s) + 1 = n_out.'
        

        # Define the filter parameters associated with the Conv2D module
        parameter_list = []
        F_shape = (D_in, filter_size, filter_size)
        for i in range(filter_count):
            F = Data(ID = ID + f'_F{i}',
                        shape = F_shape,
                        val = np.random.randn(*F_shape) * 0.01,
                        is_frozen = is_frozen,
                        optimizer_details = optimizer_details)
            
            parameter_list.append(F)

        super().__init__(ID, inputs, output, parameter_list = parameter_list,
                         learning_rate = learning_rate, is_frozen = is_frozen,
                         optimizer_details = optimizer_details)
        
        # Store filter properties
        self.filter_size = filter_size
        self.filter_count = filter_count
        self.padding = padding
        self.stride = stride

        # Maintain cache for values used in forward pass that will be used in backward pass also
        self.cache = {'X_flat': np.array(0), 'F_flat': np.array(0), 'X_shape': (), 'F_shape': (),
                      'conv_flat_shape': (), 'batch_size': 0}


    @override
    def forward(self) -> None:
        batch_size, D_in, H_in, W_in = self.inputs[0].val.shape

        F_shape = (D_in, self.filter_size, self.filter_size)
        X_shape = (D_in, H_in, W_in)

        # Create the X_flat matrix
        Xi_flat_list = []
        for m in range(batch_size):
            # Get input
            Xi = self.inputs[0].val[m]
            # Flatten Xi
            Xi_flat = im2col(Xi, self.filter_size, self.padding, self.stride)
            Xi_flat_list.append(Xi_flat)
        
        X_flat = np.stack(Xi_flat_list, axis = 0)

        # Create the F_flat matrix
        Fi_flat_list = []
        for F in self.parameter_list:
            Fi = F.val
            Fi_flat = Fi.reshape(1, -1)
            Fi_flat_list.append(Fi_flat)
        
        F_flat = np.vstack(Fi_flat_list)

        # Compute the output height and width
        o = (H_in + 2 * self.padding - self.filter_size) // self.stride + 1
        # Compute the convolution
        conv_flat = np.einsum('il,jlk->jik', F_flat, X_flat)

        # Reshape the flattened convolution into a 4D matrix of appropriate dimensions
        conv_shaped = conv_flat.reshape(-1, self.filter_count, o, o)

        # Update the output value
        self.output.val = conv_shaped

        # Cache useful values to be used in backward pass
        self.cache['conv_flat_shape'] = conv_flat.shape
        self.cache['X_flat'] = X_flat
        self.cache['F_flat'] = F_flat
        self.cache['X_shape'] = X_shape
        self.cache['F_shape'] = F_shape
        self.cache['batch_size'] = batch_size


    @override
    def backward(self) -> None:
        # Get the 4D derivative of the loss w.r.t. output
        conv_deriv_shaped = self.output.deriv
        # Reshape it to the dimensions of conv_flat as in the forward pass
        conv_flat_deriv = conv_deriv_shaped.reshape(*self.cache['conv_flat_shape'])

        # Filter derivatives
        # Find derivative of the loss w.r.t. the F_flat matrix used in forward pass
        F_flat_deriv = np.einsum('ebg,eag->ab', self.cache['X_flat'], conv_flat_deriv)
        # Get derivative for each filter
        for i in range(self.filter_count):
            self.parameter_list[i].deriv = F_flat_deriv[i].reshape(self.cache['F_shape'])
        

        # Input derivatives
        # Find derivative of the loss w.r.t. the X_flat tensor used in foward pass
        X_flat_deriv = np.einsum('fb,afc->abc', self.cache['F_flat'], conv_flat_deriv)
        # Get derivative for each training example
        Xi_deriv_list = []
        for i in range(self.cache['batch_size']):
            deriv = col2im(X_flat_deriv[i], self.cache['X_shape'], self.filter_size, self.padding, self.stride)
            Xi_deriv_list.append(deriv)
        
        self.inputs[0].deriv = np.stack(Xi_deriv_list, axis = 0)



def im2col(X, filter_size, padding, stride):
    # Pad the input
    X = np.pad(X, pad_width = ((0, 0), (padding, padding), (padding, padding)))

    # Determine the output size
    D, H, W = X.shape
    # Padding has already been added to X
    o = (H - filter_size) // stride + 1

    X_flat = np.zeros((filter_size * filter_size * D, o * o))

    col = 0
    for i in range(o):
        for j in range(o):
            idx_i, idx_j = i * stride, j * stride
            X_patch = X[:, idx_i: idx_i + filter_size, idx_j: idx_j + filter_size]
            X_flat[:, col: col + 1] = X_patch.reshape(-1, 1)
            col += 1
    
    return X_flat


def col2im(X_deriv_flat, X_shape, filter_size, padding, stride):
    D, H, W = X_shape

    # Set the derivative size
    X_deriv = np.zeros((D, H + 2 * padding, W + 2 * padding))

    o = (H + 2 * padding - filter_size) // stride + 1

    col = 0
    for i in range(o):
        for j in range(o):
            idx_i, idx_j = i * stride, j * stride
            X_deriv_flat_col = X_deriv_flat[:, col: col + 1]
            X_deriv_flat_col_shaped = X_deriv_flat_col.reshape(D, filter_size, filter_size)
            X_deriv[:, idx_i: idx_i + filter_size, idx_j: idx_j + filter_size] += X_deriv_flat_col_shaped
            col += 1
    
    # Remove derivatives introduced due to padded 0s
    X_deriv = X_deriv[:, padding: padding + H, padding: padding + W]
    
    return X_deriv