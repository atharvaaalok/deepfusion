import numpy as np
from typing import override

from .module import Module
from ..data.data import Data
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS


class Conv2D(Module):
    """Convolution 2D module that given an input X and a defined filter size and filter count
    produces an activation map via convolving the filter on the input.

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
        F_shape = (D_in, filter_size, filter_size)
        self.F = Data(ID = ID + '_F',
                      shape = F_shape,
                      val = np.random.randn(*F_shape) * 0.01,
                      is_frozen = is_frozen,
                      optimizer_details = optimizer_details)

        parameter_list = [self.F]

        super().__init__(ID, inputs, output, parameter_list = parameter_list,
                         learning_rate = learning_rate, is_frozen = is_frozen,
                         optimizer_details = optimizer_details)
        
        # Store filter properties
        self.filter_size = filter_size
        self.filter_count = filter_count
        self.padding = padding
        self.stride = stride


    @override
    def forward(self) -> None:
        D_in, H_in, W_in, batch_size = self.inputs[0].val.shape

        # Convolve the filter over each input in the mini batch
        for m in range(batch_size):
            # Get input
            X = self.inputs[0].val[..., m]
            # Convolve the filter and get the activation map
            conv = _conv2D(X, self.F.val, padding = self.padding, stride = self.stride)
        
        self.output.val = conv[..., np.newaxis]


    @override
    def backward(self) -> None:
        D_in, H_in, W_in, batch_size = self.inputs[0].val.shape

        # Get derivative w.r.t. input and filter for each input in the mini batch
        for m in range(batch_size):
            # Get input
            X = self.inputs[0].val[..., m]

            # Get output deriv
            out_deriv = self.output.deriv[..., 0]

            # Get derivative w.r.t. filter
            self.F.deriv = _conv2D(X, _dilate(out_deriv, dilation = self.stride - 1), padding = self.padding, stride = 1)

            # Get derivative w.r.t. input
            F_rotate = np.flip(self.F.val, axis = (1, 2))
            X_deriv = _conv2D(_dilate(out_deriv, dilation = self.stride - 1), F_rotate, padding = F_rotate.shape[1] - 1, stride = 1)
            X_deriv = X_deriv[:, self.padding: self.padding + H_in, self.padding: self.padding + H_in]
            self.inputs[0].deriv = X_deriv[..., np.newaxis]


def _conv2D(X, F, padding, stride):
        # Pad the input
        X = X[0, :, :]
        X = np.pad(X, padding)
        X = X[np.newaxis, :, :]

        # Determine output size
        D, H, W = X.shape
        filter_size = F.shape[1]
        o = (H - filter_size) // stride + 1
        conv = np.zeros((1, o, o))

        for i in range(o):
            for j in range(o):
                idx_i, idx_j = i * stride, j * stride
                X_patch = X[:, idx_i: idx_i + filter_size, idx_j: idx_j + filter_size]
                conv[0, i, j] = np.sum(X_patch * F)
        
        return conv


def _dilate(X, dilation):
    D, H, W = X.shape

    # New dimensions for rows and columns
    H_new = H + (H - 1) * dilation
    W_new = W + (W - 1) * dilation

    # Create array of dilated shape filled with zeros
    X_dilated = np.zeros((D, H_new, W_new))

    # Process each depth slice
    for d in range(D):
        # Fill elements of X into X_dilated at the appropriate locations
        X_dilated[d, 0: H_new: dilation + 1, 0: W_new: dilation + 1] = X[d, :, :]
    
    return X_dilated