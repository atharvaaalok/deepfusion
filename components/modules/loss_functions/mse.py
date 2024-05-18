from ....utils.backend import Backend
np = Backend.get_array_module()

from typing import override

from ..module import Module
from ...data.data import Data


class MSE(Module):
    """Mean Squared Error (MSE) module that computes the loss given predictions and target outputs.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects (predictions, target outputs) used to calculate the loss.
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

        # Go through checks first
        assert inputs[0].shape[1] == 1 and inputs[1].shape[1] == 1, \
            'For MSE both input shapes should be (B, 1).'
        assert output.shape == (1, 1), 'For MSE output shape should be (1, 1)).'

        super().__init__(ID, inputs, output)

        # Cache values during forward pass that will be useful in backward pass
        self.cache = {'h_minus_y': 0}
    
    
    @override
    def forward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        # Cache values that will be used during backward pass
        self.cache['h_minus_y'] = self.inputs[0].val - self.inputs[1].val

        self.output.val = (1 / batch_size) * (1 / 2) * np.sum((self.cache['h_minus_y']) ** 2)

        self.output.deriv = 1.0


    @override
    def backward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        # Calculate and store value that will be used for derivative of both inputs to speed compute
        deriv = (1 / batch_size) * (self.cache['h_minus_y']) * self.output.deriv

        self.inputs[0].deriv += deriv
        self.inputs[1].deriv += (-deriv)