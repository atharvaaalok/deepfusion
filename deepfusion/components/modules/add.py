from typing import override

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from .module import Module
from ..data.data import Data


class Add(Module):
    """Takes in two values and adds them.

    Args:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the modules output after transforming the input values.
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
        """Initializes Addition module based on ID, inputs and output."""

        # Go through checks first
        assert inputs[0].shape == inputs[1].shape, 'For Add both inputs should be of same shape.'
        assert inputs[0].shape == output.shape, 'For Add input and output shape should be same.'

        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        self.output.val = self.inputs[0].val + self.inputs[1].val
    

    @override
    def backward(self) -> None:
        self.inputs[0].deriv += self.output.deriv
        self.inputs[1].deriv += self.output.deriv