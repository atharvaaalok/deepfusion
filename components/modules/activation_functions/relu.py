import numpy as np
from typing import override

from ..module import Module
from ...data.data import Data


class Relu(Module):
    """Rectified Linear Unit (ReLU) module that given an input X computes max(0, X) element-wise.
    
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
        """Initializes the Relu module based on ID, inputs and output."""

        # Go through checks first
        assert inputs[0].shape == output.shape, 'For Relu input and output shape should be same.'

        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        self.output.val = np.maximum(self.inputs[0].val, 0.0)
    

    @override
    def backward(self) -> None:
        self.inputs[0].deriv = self.inputs[0].deriv + self.output.deriv * np.where(self.inputs[0].val >= 0.0, 1.0, 0.0)