import numpy as np
from typing import override

from ..module import Module
from ...data.data import Data
from ..utils.functions import _tanh, _tanh_deriv


class Tanh(Module):
    """Tanh module that given an input X computes (exp(2x) - 1) / (exp(2x) + 1) element-wise.

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
        """Initializes the Tanh module based on ID, inputs and output."""
        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        self.output.val = _tanh(self.inputs[0].val)
    

    @override
    def backward(self) -> None:
        self.inputs[0].deriv += self.output.deriv * _tanh_deriv(self.inputs[0].val)