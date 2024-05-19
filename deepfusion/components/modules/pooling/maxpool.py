from ....utils.backend import Backend
np = Backend.get_array_module()

from typing import override

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
        D_in, H_in, W_in = inputs[0].shape
        D_out, H_out, W_out = output.shape

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

        # Get the output shape and define dummy output
        o = (H_in + 2 * self.padding - self.filter_size) // self.stride + 1
        maxpool = np.zeros((batch_size, D_in, o, o))

        for m in range(batch_size):
            # Get input
            Xi = self.inputs[0].val[m]
            for d in range(D_in):
                # Take a depth slice of Xi
                Xid = Xi[d, :, :]
                # Move the filter over this depth slice
                for i in range(o):
                    for j in range(o):
                        idx_i, idx_j = i * self.stride, j * self.stride
                        # Get a patch
                        Xid_patch = Xid[idx_i: idx_i + self.filter_size, idx_j: idx_j + self.filter_size]
                        maxpool[m, d, i, j] = np.max(Xid_patch)
        

        self.output.val = maxpool


    @override
    def backward(self) -> None:
        batch_size, D_in, H_in, W_in = self.inputs[0].val.shape

        # Get output derivative
        out_deriv = self.output.deriv

        # Get the output shape and define dummy output
        o = (H_in + 2 * self.padding - self.filter_size) // self.stride + 1

        # Create dummy input derivative
        in_deriv = np.zeros((batch_size, D_in, H_in, W_in))

        # Move the filters and set the values for input derivative
        for m in range(batch_size):
            # Get input
            Xi = self.inputs[0].val[m]
            for d in range(D_in):
                # Take a depth slice of Xi
                Xid = Xi[d, :, :]
                # Move the filter over this depth slice
                for i in range(o):
                    for j in range(o):
                        idx_i, idx_j = i * self.stride, j * self.stride
                        # Get a patch
                        Xid_patch = Xid[idx_i: idx_i + self.filter_size, idx_j: idx_j + self.filter_size]
                        max_idx = np.argwhere(Xid_patch == np.max(Xid_patch))
                        i_max, j_max = max_idx[0]

                        # Set the derivative at max location to output derivative value
                        in_deriv[m, d, idx_i + i_max, idx_j + j_max] = out_deriv[m, d, i, j]
        
        
        self.inputs[0].deriv += in_deriv