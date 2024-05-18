from ....utils.backend import Backend
np = Backend.get_array_module()

from typing import Optional, override

from ..module import Module
from ...data.data import Data
from ...optimizers import DEFAULT_OPTIMIZER_DETAILS
from ..utils.functions import _sigmoid, _sigmoid_deriv


class RNN1(Module):
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
        """Initializes RNN1 module based on ID, inputs, output and other optional parameters."""

        # Go through checks first
        assert inputs[0].shape == output.shape, \
            'For RNN1 input and output activation shape should be same.'
        
        # Define the weight and bias parameters associated with the RNN1 module
        Waa_shape = (inputs[0].shape[1], output.shape[1])
        Wax_shape = (inputs[1].shape[1], output.shape[1])
        ba_shape = (output.shape)

        self.Waa = Data(ID = ID + '_Waa', shape = Waa_shape,
                        val = _weight_initialization(weight_init_type, Waa_shape),
                        is_frozen = is_frozen, optimizer_details = optimizer_details)
        
        self.Wax = Data(ID = ID + '_Wax', shape = Wax_shape,
                        val = _weight_initialization(weight_init_type, Wax_shape),
                        is_frozen = is_frozen, optimizer_details = optimizer_details)
        
        self.ba = Data(ID = ID + '_ba', shape = ba_shape, val = np.zeros(ba_shape),
                       is_frozen = is_frozen, optimizer_details = optimizer_details)
        
        parameter_list = [self.Waa, self.Wax, self.ba]

        super().__init__(ID, inputs = inputs, output = output, parameter_list = parameter_list,
                         learning_rate = learning_rate, is_frozen = is_frozen,
                         optimizer_details = optimizer_details)
        
        # Cache values calculated in the forward pass that will be useful in the backward pass
        self.cache = {'temp_a': 0}
    

    @override
    def forward(self) -> None:
        temp_a = self.inputs[0].val @ self.Waa.val + self.inputs[1].val @ self.Wax.val + self.ba.val
        a1 = _sigmoid(temp_a)

        self.output.val = a1

        self.cache['temp_a'] = temp_a


    @override
    def backward(self) -> None:
        d_temp_a = _sigmoid_deriv(self.cache['temp_a']) * self.output.deriv

        self.inputs[0].deriv += d_temp_a @ self.Waa.val.T
        self.inputs[1].deriv += d_temp_a @ self.Wax.val.T

        self.ba.deriv += np.sum(d_temp_a, axis = 0, keepdims = True)
        self.Waa.deriv += self.inputs[0].val.T @ d_temp_a
        self.Wax.deriv += self.inputs[1].val.T @ d_temp_a



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