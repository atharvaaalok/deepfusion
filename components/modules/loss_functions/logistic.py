from ....utils.backend import Backend
np = Backend.get_array_module()

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

        # Go through checks first
        assert inputs[0].shape[1] == 1 and inputs[1].shape[1] == 1, \
            'For MSE both input shapes should be (B, 1).'
        assert output.shape == (1, 1), 'For Logistic output shape should be (1, 1).'

        super().__init__(ID, inputs, output)

        # Cache values during forward pass that will be useful in backward pass
        self.cache = {'sigmoid_t': 0}
    

    @override
    def forward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        t = self.inputs[0].val
        y = self.inputs[1].val

        # Cache values that will be used during backward pass
        sig_t = _sigmoid(t)
        self.cache['sigmoid_t'] = sig_t

        # self.output.val = (1 / batch_size) * np.sum(-y * np.log(_sigmoid(t)) - (1 - y) * np.log(_sigmoid(-t)))

        # The following is a computationally efficient way of calculating the above expression
        # Note that sigmoid(-t) = 1 - sigmoid(t)
        self.output.val = (1 / batch_size) * np.sum(-y * np.log(sig_t) - (1 - y) * np.log(1 - sig_t))

        self.output.deriv = 1.0


    @override
    def backward(self) -> None:
        batch_size = self.inputs[0].val.shape[0]

        t = self.inputs[0].val
        y = self.inputs[1].val

        sig_t = self.cache['sigmoid_t']

        # self.inputs[0].deriv += (1 / batch_size) * (-y * _sigmoid(-t) + (1 - y) * _sigmoid(t)) * self.output.deriv

        # The following is a computationally efficient way of calculating the above expression
        self.inputs[0].deriv += (1 / batch_size) * (-y * (1 - sig_t) + (1 - y) * sig_t) * self.output.deriv
        self.inputs[1].deriv = 0