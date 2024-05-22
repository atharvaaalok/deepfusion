from typing_extensions import override

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from ..module import Module
from ...data.data import Data
from ..utils.functions import _sigmoid, _sigmoid_deriv


class Sigmoid(Module):
    """Sigmoid module that given an input X computes 1 / (1 + exp(-X)) element-wise.

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
        """Initializes the Sigmoid module based on ID, inputs and output."""

        # Go through checks first
        assert inputs[0].shape == output.shape, 'For Sigmoid input and output shape should be same.'

        super().__init__(ID, inputs, output)

        # Cache values during forward pass that will be useful in backward pass
        self.cache = {'sigmoid_x': 0}
    

    @override
    def forward(self) -> None:
        # Cache sigmoid_x to use during backward pass
        self.cache['sigmoid_x'] = _sigmoid(self.inputs[0].val)

        self.output.val = self.cache['sigmoid_x']
    

    @override
    def backward(self) -> None:
        self.inputs[0].deriv += self.output.deriv * _sigmoid_deriv(sigmoid_x = self.cache['sigmoid_x'])