from ...utils.backend import Backend
np = Backend.get_array_module()

from typing import override

from .module import Module
from ..data.data import Data


class Flatten(Module):
    """Flatten takes any tensor of shape (m, a, b, c, ...) where m is batch size and converts it
    into a 2D matrix of size (m, a * b * c * ...).

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
    """

    different_at_train_test = False
    is_regularizable = False


    def __init__(self, ID: str, inputs: list[Data], output: Data) -> None:
        """Initializes Flatten module based on ID, inputs and output."""

        # Go through checks first
        total_input_elements = np.prod(inputs[0].shape)
        total_output_elements = output.shape[1]
        assert total_input_elements == total_output_elements, \
            'For Flatten module the total elements in input should equal the total output elements.'

        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        # Flatten the input into a 2D matrix of size (m, dim1) where dim1 is no. of elements in
        # each training example and m is the batch size
        self.output.val = self.inputs[0].val.reshape(batch_size, -1)


    @override
    def backward(self) -> None:
        # Backward pass is simply reshaping the output derivative
        self.inputs[0].deriv = self.output.deriv.reshape(self.inputs[0].val.shape)