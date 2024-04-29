import numpy as np


def get_regularizer_with_deriv(regularizer_name):

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