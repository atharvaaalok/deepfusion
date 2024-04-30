import numpy as np

from .module import Module
from ..data.data import Data
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS


class MatMul(Module):

    different_at_train_test = False
    is_regularizable = True


    def __init__(self, ID, inputs, output, learning_rate = 1e-6, is_frozen = False, optimizer_details = DEFAULT_OPTIMIZER_DETAILS, is_regularized = False, regularizer_details = None, weight_init_type = 'He'):        

        if is_regularized and regularizer_details is None:
            raise ValueError('If Module is regularized, regularizer details must be provided.')
        
        self.is_regularized = is_regularized
        self.regularizer_details = regularizer_details if is_regularized else None
        
        W_shape = (output.shape[0], inputs[0].shape[0])
        b_shape = (output.shape)
        self.W = Data(ID = ID + '_W', shape = W_shape, val = _weight_initialization(weight_init_type, W_shape), is_frozen = is_frozen, optimizer_details = optimizer_details, is_regularized = is_regularized, regularizer_details = regularizer_details)
        self.b = Data(ID = ID + '_b', shape = b_shape, val = np.zeros(b_shape), is_frozen = is_frozen, optimizer_details = optimizer_details)

        parameter_list = [self.W, self.b]

        super().__init__(ID, inputs, output, parameter_list = parameter_list, learning_rate = learning_rate, is_frozen = is_frozen, optimizer_details = optimizer_details)
    

    def forward(self):
        self.output.val = self.W.val @ self.inputs[0].val + self.b.val

    
    def backward(self):
        batch_size = self.inputs[0].val.shape[1]

        self.inputs[0].deriv = self.W.val.T @ self.output.deriv

        self.W.deriv = (1 / batch_size) * (self.output.deriv @ self.inputs[0].val.T)
        self.b.deriv = (1 / batch_size) * (np.sum(self.output.deriv, axis = 1, keepdims = True))

    
    def set_regularization(self, regularizer_details):
        self.is_regularized = True
        self.regularizer_details = regularizer_details
        self.W.set_regularization(regularizer_details)


def _weight_initialization(weight_init_type, W_shape):
    
    available_init = ['He']

    match weight_init_type:
        case 'He':
            return np.random.randn(*W_shape) * np.sqrt(2 / W_shape[1])
        
        case _:
            raise ValueError(f'Specified initialization method not available. Choose from - {available_init}')