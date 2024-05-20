from deepfusion.utils.backend import Backend
np = Backend.get_array_module()

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

        # Go through checks first
        assert inputs[0].shape == output.shape, 'For Tanh input and output shape should be same.'

        super().__init__(ID, inputs, output)

        # Cache values during forward pass that will be useful in backward pass
        self.cache = {'tanh_x': 0}
    

    @override
    def forward(self) -> None:
        # Cache tanh_x to use during backward pass
        self.cache['tanh_x'] = _tanh(self.inputs[0].val)

        self.output.val = self.cache['tanh_x']
    

    @override
    def backward(self) -> None:
        self.inputs[0].deriv += self.output.deriv * _tanh_deriv(tanh_x = self.cache['tanh_x'])