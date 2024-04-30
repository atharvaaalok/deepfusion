import numpy as np
from typing import Optional, override

from .module import Module
from ..data.data import Data
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS


class MatMul(Module):
    """Matrix Multiplication module that given an input X, outputs WX + b.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the modules output after transforming the input values.
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
        is_regularized:
            Boolean that decides if the weight parameter for the module is regularized.
        regularizer_details:
            Dictionary containing the regularization strength and the regularizer name.
        W:
            Weight matrix parameter for the matrix multiplication.
        b:
            Bias vector parameter for the module.
    """

    different_at_train_test = False
    is_regularizable = True


    def __init__(
        self,
        ID: str,
        inputs: list[Data],
        output: Data,
        learning_rate: float = 1e-6,
        is_frozen: bool = False,
        optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS,
        is_regularized: bool = False,
        regularizer_details: Optional[dict] = None,
        weight_init_type: str = 'He',
    ) -> None:
        """Initializes Matrix Multiplication module based on ID, inputs, output and other optional
        parameters.
        
        Raises:
            ValueError: If data is regularized but regularizer details are not provided.
            ValueError: If input is not a list of Data objects or the output is not a Data object.
        """

        if is_regularized and regularizer_details is None:
            raise ValueError('If Module is regularized, regularizer details must be provided.')
        
        self.is_regularized = is_regularized
        self.regularizer_details = regularizer_details if is_regularized else None
        
        # Define the weight (W) and bias (b) parameters associated with the MatMul module
        W_shape = (output.shape[0], inputs[0].shape[0])
        b_shape = (output.shape)
        self.W = Data(ID = ID + '_W',
                      shape = W_shape,
                      val = _weight_initialization(weight_init_type, W_shape),
                      is_frozen = is_frozen,
                      optimizer_details = optimizer_details,
                      is_regularized = is_regularized,
                      regularizer_details = regularizer_details)
        
        self.b = Data(ID = ID + '_b',
                      shape = b_shape,
                      val = np.zeros(b_shape),
                      is_frozen = is_frozen,
                      optimizer_details = optimizer_details)

        parameter_list = [self.W, self.b]

        super().__init__(ID, inputs, output, parameter_list = parameter_list,
                         learning_rate = learning_rate, is_frozen = is_frozen,
                         optimizer_details = optimizer_details)
    

    @override
    def forward(self) -> None:
        # Output = WX + b
        self.output.val = self.W.val @ self.inputs[0].val + self.b.val

    
    @override
    def backward(self) -> None:
        batch_size = self.inputs[0].val.shape[1]

        self.inputs[0].deriv = self.W.val.T @ self.output.deriv

        self.W.deriv = (1 / batch_size) * (self.output.deriv @ self.inputs[0].val.T)
        self.b.deriv = (1 / batch_size) * (np.sum(self.output.deriv, axis = 1, keepdims = True))

    
    def set_regularization(self, regularizer_details: dict) -> None:
        """Regularize appropriate parameters of the module using specified regularization strength
        and function.
        
        Args:
            regularizer_details: Dictionary with regularizer strength and function name.
        """
        self.is_regularized = True
        self.regularizer_details = regularizer_details
        self.W.set_regularization(regularizer_details)


def _weight_initialization(weight_init_type: str, W_shape: tuple[int, int]) -> None:
    """Utility function to initialize and return a weight matrix based on specified initialization
    method.

    Args:
        weight_init_type: String chosen from the available initialization choices.
        W_shape: Shape of the weight matrix to be initialized.

    Raises:
        ValueError: If the specified weight initialization method is not available.
    """
    
    available_init = ['He']

    match weight_init_type:
        case 'He':
            return np.random.randn(*W_shape) * np.sqrt(2 / W_shape[1])
        
        case _:
            raise ValueError('Specified initialization method not available.' \
                             f' Choose from - {available_init}')