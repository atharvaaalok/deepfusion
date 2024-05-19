from ....utils.backend import Backend
np = Backend.get_array_module()

from typing import override

from ..module import Module
from ...data.data import Data


class SumLoss(Module):
    """Sum loss module that computes the loss as the sum of all the input values.

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

        # Go through the checks first
        assert output.shape == (1, 1), 'For Sum loss module output shape should be (1, 1).'

        super().__init__(ID, inputs, output)
    

    @override
    def forward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        self.output.val = (1 / batch_size) * np.sum(self.inputs[0].val)

        self.output.deriv = np.array([1.0])
    

    @override
    def backward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        self.inputs[0].deriv += (1 / batch_size) * np.ones(self.inputs[0].val.shape) * self.output.deriv