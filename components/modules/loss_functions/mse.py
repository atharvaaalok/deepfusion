import numpy as np
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
        assert inputs[0].shape == (1, 1) and inputs[1].shape == (1, 1), \
            'For MSE both input shapes should be (1, 1).'
        assert output.shape == (1, 1), 'For MSE output shape should be (1, 1).'

        super().__init__(ID, inputs, output)
    
    
    @override
    def forward(self) -> None:
        batch_size = self.inputs[0].val.shape[1]
        self.output.val = (1 / batch_size) * (1 / 2) * np.sum((self.inputs[0].val - self.inputs[1].val) ** 2)

        self.output.deriv = 1.0


    @override
    def backward(self) -> None:
        batch_size = self.inputs[0].val.shape[1]

        self.inputs[0].deriv = self.inputs[0].deriv + (1 / batch_size) * (self.inputs[0].val - self.inputs[1].val) * self.output.deriv
        self.inputs[1].deriv = self.inputs[1].deriv + (1 / batch_size) * (self.inputs[1].val - self.inputs[0].val) * self.output.deriv