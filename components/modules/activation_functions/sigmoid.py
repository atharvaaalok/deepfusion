import numpy as np
from typing import override
import numpy.typing as npt

from ..module import Module
from ...data.data import Data


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
        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        self.output.val = _sigmoid(self.inputs[0].val)
    

    @override
    def backward(self) -> None:
        self.inputs[0].deriv = self.output.deriv * _sigmoid_deriv(self.inputs[0].val)
    

def _sigmoid(x: npt.NDArray) -> npt.NDArray:
    """Sigmoid function that given an input X computes 1 / (1 + exp(-X)) element-wise.

    Calculations are done differently for positive and negative x values to prevent overflow
    situations. This is because for large negative X values exp(-X) is a very large positive
    value and overflow warning is thrown.
    
    Args:
        x: Input tensor for which the function is to be applied element-wise.
    """

    def _positive_sigmoid(x):
        """Utility function that calculates sigmoid for positive x values."""
        return 1 / (1 + np.exp(-x))
    
    def _negative_sigmoid(x):
        """Utility function that calculates sigmoid for negative x values."""
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)
    
    # Calculate sigmoid separately for positive and negative x values
    # Interesting details can be found in this stackoverflow answer:
    # https://stackoverflow.com/a/64717799/12842649
    pos_x_mask = x >= 0
    neg_x_mask = ~pos_x_mask

    sigmoid_x = np.zeros(x.shape)
    sigmoid_x[pos_x_mask] = _positive_sigmoid(x[pos_x_mask])
    sigmoid_x[neg_x_mask] = _negative_sigmoid(x[neg_x_mask])

    return sigmoid_x


def _sigmoid_deriv(x: npt.NDArray) -> npt.NDArray:
    """Derivative of the sigmoid function.
    
    Args:
        x: Input tensor at which the derivative of the sigmoid function is to be calculated.
    """

    sigmoid_x = _sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)