import numpy as np
from typing import override

from ..module import Module
from ...data.data import Data


class LRelu(Module):
    """Leaky Rectified Linear Unit (LReLU) module that given an input X and hyperparameter alpha
    computes (X if X >= 0 else alpha * X) element-wise.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the module's output after transforming the input values.
        alpha:
            Hyperparameter that is used in computing (X if X >= 0 else alpha * X) for given input X.
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


    def __init__(self, ID: str, inputs: list[Data], output: Data, alpha: float = 0.01) -> None:
        """Initializes the LRelu module based on ID, inputs, output and hyperparameter alpha."""
        super().__init__(ID, inputs, output)

        self.alpha = alpha
    

    @override
    def forward(self) -> None:
        x = self.inputs[0].val
        self.output.val = np.where(x >= 0, x, self.alpha * x)
    

    @override
    def backward(self) -> None:
        self.inputs[0].deriv += self.output.deriv * np.where(self.inputs[0].val >= 0, 1, self.alpha)