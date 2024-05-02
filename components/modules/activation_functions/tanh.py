import numpy as np
from typing import override
import numpy.typing as npt

from ..module import Module
from ...data.data import Data


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
        self.inputs[0].deriv = self.output.deriv * _tanh_deriv(self.inputs[0].val)


def _tanh(x: npt.NDArray) -> npt.NDArray:
    """Tanh function that given an input X computes (exp(2x) - 1) / (exp(2x) + 1) element-wise.

    Calculations are done differently for positive and negative x values to prevent overflow
    situations. This is because for large positive X values exp(X) is a very large positive value
    and overflow warning is thrown.
    
    Args:
        x: Input tensor for which the function is to be applied element-wise.
    """

    def _positive_tanh(x):
        """Utility function that calculates tanh for positive x values."""
        exp_neg_2x = np.exp(-2 * x)
        return (1 - exp_neg_2x) / (1 + exp_neg_2x)
    
    def _negative_tanh(x):
        """Utility function that calculates tanh for negative x values."""
        exp_2x = np.exp(2 * x)
        return (exp_2x - 1) / (exp_2x + 1)

    # Calculate tanh separately for positive and negative x values
    # Logic for splitting as such is similar to that explained for the sigmoid function
    # Interesting details (for sigmoid) can be found in this stackoverflow answer:
    # https://stackoverflow.com/a/64717799/12842649
    pos_x_mask = x >= 0
    neg_x_mask = ~pos_x_mask

    tanh_x = np.zeros(x.shape)
    tanh_x[pos_x_mask] = _positive_tanh(x[pos_x_mask])
    tanh_x[neg_x_mask] = _negative_tanh(x[neg_x_mask])

    return tanh_x


def _tanh_deriv(x: npt.NDArray) -> npt.NDArray:
    """Derivative of the tanh function.
    
    Args:
        x: Input tensor at which the derivative of the tanh function is to be calculated.
    """

    tanh_x = _tanh(x)
    return 1 - tanh_x ** 2