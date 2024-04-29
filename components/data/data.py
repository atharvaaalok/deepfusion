import numpy as np

from .regularizers import Regularizer
from ..optimizers import get_optimizer, DEFAULT_OPTIMIZER_DETAILS

class Data:

    def __init__(self, ID, shape, val = None, is_frozen = True, optimizer_details = DEFAULT_OPTIMIZER_DETAILS, learning_rate = 1e-6, is_regularized = False, regularizer_details = None):
        self.ID = ID

        self.shape = shape
        self.val = np.zeros(shape) if val is None else val
        self.deriv = np.zeros(1)

        self.is_frozen = is_frozen

        self.optimizer = get_optimizer(shape, optimizer_details) if not is_frozen else None

        self.learning_rate = learning_rate

        self.is_regularized = is_regularized
        if is_regularized and regularizer_details is None:
            raise ValueError('If Data is regularized, regularizer details must be provided.')
        self.regularizer = Regularizer(regularizer_details) if regularizer_details is not None else None
    
    
    def update(self):
        if not self.is_frozen:
            optimizer_step = self.optimizer.step(self.deriv)
            regularizer_step = self.regularizer.reg_fn_deriv(self.val) if self.is_regularized else 0
            self.val -= self.learning_rate * (optimizer_step + self.regularizer.reg_strength * regularizer_step)


    def freeze(self):
        self.is_frozen = True
        self.optimizer = None


    def unfreeze(self, optimizer_details = DEFAULT_OPTIMIZER_DETAILS):
        self.is_frozen = False
        self.optimizer = get_optimizer(self.shape, optimizer_details)


    def clear_grads(self):
        self.deriv = np.zeros(1)


    def set_regularization(self, regularizer_details):
        self.is_regularized = True
        self.regularizer = Regularizer(regularizer_details)


    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate