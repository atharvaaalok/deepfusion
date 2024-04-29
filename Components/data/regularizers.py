import numpy as np


class Regularizer:

    def __init__(self, regularizer_details):
        self.reg_strength = regularizer_details['reg_strength']
        self.reg_name = regularizer_details['reg_name']
        self.reg_fn, self.reg_fn_deriv = _get_regularizer_with_deriv(self.reg_name)


def _get_regularizer_with_deriv(regularizer_name):

    try:
        return _regularizers_with_deriv_dict[regularizer_name]
    except KeyError:
        raise ValueError(f'Specified regularizer not available. Choose from - {list(_regularizers_with_deriv_dict)}')



def _L2_regularizer(x):
    return np.sum(x ** 2)


def _L2_regularizer_deriv(x):
    return 2 * x


_regularizers_with_deriv_dict = {
    'L2': (_L2_regularizer, _L2_regularizer_deriv),
}