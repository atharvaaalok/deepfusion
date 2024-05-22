from typing_extensions import override

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from ..module import Module
from ...data.data import Data


class MaxPool(Module):
    """Max Pool module that given an input X and a defined filter size produces an activation map by
    taking the max over each filter position on the input.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the module's output after transforming the input values.
        parameter_list:
            Not used by the module. Set to default value by the Module base class.
        learning_rate:
            Not used by the module. Set to default value by the Module base class.
        is_frozen:
            Not used by the module. Set to default value by the Module base class.
        optimizer_details:
            Not used by the module. Set to default value by the Module base class.
        filter_size:
            Height and width of the filter that is convolved over the input.
        padding:
            Amount of 0 padding to the edges of the input.
        stride:
            Step size of the filter over the input.
    """

    different_at_train_test = False
    is_regularizable = False


    def __init__(
        self,
        ID: str,
        inputs: list[Data],
        output: Data,
        filter_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """Initializes the MaxPool module based on ID, inputs and output."""

        # Get input and output dimensions
        D_in, H_in, W_in = inputs[0].shape[-3:]
        D_out, H_out, W_out = output.shape[-3:]

        # Go through checks first
        assert D_in == D_out, 'MaxPool output depth should be same as input depth.'
        assert H_in == W_in, 'MaxPool input should be square.'
        assert H_out == W_out, 'MaxPool output should be square.'
        assert (H_in + 2 * padding - filter_size) % stride == 0, \
            'MaxPool input size, filter size, padding and stride must satisfy (n + 2p - f) % s = 0.'
        assert ((H_in + 2 * padding - filter_size) / stride) + 1 == H_out, \
            'MaxPool output size should be consistent with ((n_in + 2p - f) / s) + 1 = n_out.'

        super().__init__(ID, inputs, output)

        # Store filter properties
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
    

    @override
    def forward(self) -> None:
        batch_size, D_in, H_in, W_in = self.inputs[0].val.shape

        # Apply padding if needed
        if self.padding > 0:
            X = np.pad(self.inputs[0].val, pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            X = self.inputs[0].val

        # Get the output shape and define dummy output
        o = (H_in + 2 * self.padding - self.filter_size) // self.stride + 1
        maxpool = np.zeros((batch_size, D_in, o, o))

        # Perform max pooling using vectorized operations
        for i in range(o):
            for j in range(o):
                idx_i, idx_j = i * self.stride, j * self.stride
                # Apply max pooling over this patch
                maxpool[:, :, i, j] = np.max(X[:, :, idx_i: idx_i + self.filter_size, idx_j: idx_j + self.filter_size], axis = (2, 3))
        
        self.output.val = maxpool


    @override
    def backward(self) -> None:
        batch_size, D_in, H_in, W_in = self.inputs[0].val.shape

        # Apply padding if needed
        if self.padding > 0:
            X = np.pad(self.inputs[0].val, pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            X = self.inputs[0].val

        # Get output derivative
        out_deriv = self.output.deriv

        # Get the output shape and define dummy output
        o = (H_in + 2 * self.padding - self.filter_size) // self.stride + 1

        # Create dummy input derivative
        in_deriv = np.zeros(X.shape)

        # Perform backpropagation using vectorized operations
        for i in range(o):
            for j in range(o):
                idx_i, idx_j = i * self.stride, j * self.stride
                # Get a depth patch
                X_patch = X[:, :, idx_i: idx_i + self.filter_size, idx_j: idx_j + self.filter_size]
                max_X_patch = np.max(X_patch, axis = (2, 3), keepdims = True)
                # Get derivative mask
                mask = (X_patch == max_X_patch)
                # Set input derivative
                in_deriv[:, :, idx_i: idx_i + self.filter_size, idx_j: idx_j + self.filter_size] += mask * out_deriv[:, :, i, j][:, :, None, None]
        

        # Remove the derivatives of the padded portions of the input
        if self.padding > 0:
            in_deriv = in_deriv[:, :, self.padding:-self.padding, self.padding: -self.padding]
        

        self.inputs[0].deriv += in_deriv