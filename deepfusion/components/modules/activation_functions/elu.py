from typing_extensions import override

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from ..module import Module
from ...data.data import Data
from ...optimizers import DEFAULT_OPTIMIZER_DETAILS


class ELU(Module):
    """Exponential Linear Unit (ELU) module, given an input X and learnable parameter alpha computes
    (X if X >= 0 else alpha * (exp(X) - 1)) element-wise and learns alpha in backpropagation.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the module's output after transforming the input values.
        alpha:
            Learnable parameter that is used in computing (X if X >= 0 else alpha * (exp(X) - 1))
            for given input X.
        parameter_list:
            List of parameter Data objects associated with the module. Empty if module has no
            parameters.
        learning_rate:
            Controls the step size of the update for the module's parameters.
        is_frozen:
            Boolean that decides if updates will be made to module's parameters or not.
        optimizer_details:
            Dictionary containing name of the optimizer and a dictionary of its associated
            hyperparameters.
    """

    different_at_train_test = False
    is_regularizable = False


    def __init__(
        self,
        ID: str,
        inputs: list[Data],
        output: Data,
        alpha: float = 1.0,
        learning_rate: float = 1e-6,
        is_frozen: bool = False,
        optimizer_details = DEFAULT_OPTIMIZER_DETAILS,
    ) -> None:
        """Initializes the ELU module based on ID, inputs, output, initial value for alpha and other
        optional parameters."""

        # Go through checks first
        assert inputs[0].shape == output.shape, 'For ELU input and output shape should be same.'
        
        self.alpha = Data(ID = ID + '_alpha',
                          shape = (),
                          val = np.array(alpha),
                          is_frozen = is_frozen,
                          optimizer_details = optimizer_details)

        parameter_list = [self.alpha]

        super().__init__(ID, inputs, output, parameter_list = parameter_list,
                         learning_rate = learning_rate, is_frozen = is_frozen,
                         optimizer_details = optimizer_details)
        
        # Cache values during forward pass that will be useful in backward pass
        self.cache = {'exp_x': 0}
    

    @override
    def forward(self) -> None:
        x = self.inputs[0].val
        # Cache values to be used during backward pass
        self.cache['exp_x'] = np.exp(x)
        self.output.val = np.where(x >= 0.0, x, self.alpha.val * (self.cache['exp_x'] - 1.0))
    

    @override
    def backward(self) -> None:
        x = self.inputs[0].val
        exp_x = self.cache['exp_x']
        out_deriv = self.output.deriv

        self.inputs[0].deriv += out_deriv * np.where(x >= 0.0, 1.0, self.alpha.val * exp_x)
        self.alpha.deriv += np.sum(out_deriv * np.where(x >= 0.0, 0.0, (exp_x - 1.0)))