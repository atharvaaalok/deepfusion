import numpy as np

from .regularizers import Regularizer
from ..optimizers import get_optimizer

class Data:

    def __init__(self, ID, shape, val = None, is_frozen = True, optimizer_details = None, learning_rate = 1e-6, is_regularized = False, regularizer_details = None):
        self.ID = ID

        self.shape = shape
        self.val = np.zeros(shape) if val is None else val
        self.deriv = np.zeros(shape)

        self.is_frozen = is_frozen

        if not is_frozen and optimizer_details is None:
            raise ValueError('If Data is not frozen, optimizer details must be provided.')
        self.optimizer = get_optimizer(shape, optimizer_details) if optimizer_details is not None else None

        self.learning_rate = learning_rate

        self.is_regularized = is_regularized
        if is_regularized and regularizer_details is None:
            raise ValueError('If Data is regularized, regularizer details must be provided.')
        self.regularizer = Regularizer(regularizer_details) if regularizer_details is not None else None