from ....utils.backend import Backend
np = Backend.get_array_module()

from typing import Optional, override

from ..module import Module
from ...data.data import Data
from ...optimizers import DEFAULT_OPTIMIZER_DETAILS
from ..utils.functions import _sigmoid, _sigmoid_deriv


class RNN2(Module):
    """
    """

    different_at_train_test = False
    is_regularizable = False


    def __init__(
        self,
        ID: str,
        inputs: list[Data],
        output: Data,
        learning_rate: float = 1e-6,
        is_frozen: bool = False,
        optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS,
        weight_init_type: str = 'He',
    ) -> None:
        """Initializes RNN2 module based on ID, inputs, output and other optional parameters."""

        # Define the weight and bias parameters associated with the RNN2 module
        Way_shape = (inputs[0].shape[1], output.shape[1])
        by_shape = (output.shape)

        self.Way = Data(ID = ID + '_Way', shape = Way_shape,
                        val = _weight_initialization(weight_init_type, Way_shape),
                        is_frozen = is_frozen, optimizer_details = optimizer_details)
        
        self.by = Data(ID = ID + '_by', shape = by_shape, val = np.zeros(by_shape),
                       is_frozen = is_frozen, optimizer_details = optimizer_details)
        
        parameter_list = [self.Way, self.by]

        super().__init__(ID, inputs = inputs, output = output, parameter_list = parameter_list,
                         learning_rate = learning_rate, is_frozen = is_frozen,
                         optimizer_details = optimizer_details)
        
        # Cache values calculated in the forward pass that will be useful in the backward pass
        self.cache = {'temp_y': 0}
    

    @override
    def forward(self) -> None:
        temp_y = self.inputs[0].val @ self.Way.val + self.by.val
        y = _sigmoid(temp_y)

        self.output.val = y

        self.cache['temp_y'] = temp_y


    @override
    def backward(self) -> None:
        d_temp_y = _sigmoid_deriv(self.cache['temp_y']) * self.output.deriv

        self.by.deriv = self.by.deriv + np.sum(d_temp_y, axis = 0, keepdims = True)
        self.Way.deriv = self.Way.deriv + self.inputs[0].val.T @ d_temp_y

        self.inputs[0].deriv = self.inputs[0].deriv + d_temp_y @ self.Way.val.T



def _weight_initialization(weight_init_type: str, W_shape: tuple[int, int]) -> None:
    """Utility function to initialize and return a weight matrix based on specified initialization
    method.

    Args:
        weight_init_type: String chosen from the available initialization choices.
        W_shape: Shape of the weight matrix to be initialized.

    Raises:
        ValueError: If the specified weight initialization method is not available.
    """
    
    available_init = ['Zero', 'Random', 'Xavier', 'He', 'Sparse']

    match weight_init_type:
        case 'Zero':
            return np.zeros(W_shape)
        
        case 'Random':
            return np.random.randn(*W_shape) * 0.01
        
        case 'Xavier':
            return np.random.randn(*W_shape) * np.sqrt(1 / W_shape[0])
        
        case 'He':
            return np.random.randn(*W_shape) * np.sqrt(2 / W_shape[0])
        
        case 'Sparse':
            sparsity = 0.1
            weights = np.zeros(W_shape)
            total_weights = np.prod(W_shape)
            non_zero_count = int(total_weights * sparsity)
            non_zero_indices = np.random.choice(total_weights, non_zero_count, replace = False)
            non_zero_values = 0.01 * np.random.randn(non_zero_count)
            np.put(weights, non_zero_indices, non_zero_values)
            return weights
        
        case _:
            raise ValueError('Specified initialization method not available.' \
                             f' Choose from - {available_init}')