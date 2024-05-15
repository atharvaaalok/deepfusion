import numpy as np
from typing import override

from ..module import Module
from ...data.data import Data
from ..utils.functions import _softmax


class CrossEntropy(Module):
    """Cross Entropy loss module that computes the loss given predictions and target outputs.

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
        assert inputs[0].shape == inputs[1].shape, \
            'For Cross Entropy both input shapes should be same.'
        assert output.shape == (1, 1), 'For Cross Entropy output shape should be (1, 1).'

        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        t = self.inputs[0].val
        y = self.inputs[1].val

        cross_entropy = -np.log(np.sum(_softmax(t) * y, axis = 1, keepdims = True))
        self.output.val = (1 / batch_size) * np.sum(cross_entropy)

        self.output.deriv = 1.0


    @override
    def backward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        t = self.inputs[0].val
        y = self.inputs[1].val

        self.inputs[0].deriv = self.inputs[0].deriv + (1 / batch_size) * (_softmax(t) - y) * self.output.deriv
        self.inputs[1].deriv = 0