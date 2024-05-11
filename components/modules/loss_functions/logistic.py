import numpy as np
from typing import override

from ..module import Module
from ...data.data import Data
from ..utils.functions import _sigmoid


class Logistic(Module):
    """Logistic loss module that computes the loss given predictions and target outputs.

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
        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        batch_size = self.inputs[0].val.shape[1]

        t = self.inputs[0].val
        y = self.inputs[1].val
        self.output.val = (1 / batch_size) * np.sum(-y * np.log(_sigmoid(t)) - (1 - y) * np.log(_sigmoid(-t)))

        self.output.deriv = 1.0


    @override
    def backward(self) -> None:
        batch_size = self.inputs[0].val.shape[1]

        t = self.inputs[0].val
        y = self.inputs[1].val
        self.inputs[0].deriv += (1 / batch_size) * (-y * _sigmoid(-t) + (1 - y) * _sigmoid(t)) * self.output.deriv
        self.inputs[1].deriv += 0