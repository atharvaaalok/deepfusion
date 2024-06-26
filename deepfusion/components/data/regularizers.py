from typing import Callable

import numpy.typing as npt

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()


class Regularizer:
    """Object holding the regularization strength and function and its derivative.

    Attributes:
        reg_strength: Regularization strength that scales the contribution to the loss.
        reg_name: Name of the regularizer to be used. Eg: 'L2'
        reg_fn: Regularization function.
        reg_fn_deriv: Derivative of the regularization function.
    """

    def __init__(self, regularizer_details: dict) -> None:
        self.reg_strength = regularizer_details['reg_strength']
        self.reg_name = regularizer_details['reg_name']
        self.reg_fn, self.reg_fn_deriv = _get_regularizer_with_deriv(self.reg_name)
    

    def __str__(self) -> str:
        print_regularizer = f'Name: {self.reg_name}, Strength: {self.reg_strength:0.2e}'
        return print_regularizer



def _get_regularizer_with_deriv(regularizer_name: str) -> tuple[Callable, Callable]:
    """For the specified regularizer returns the regularization function and its derivative.
    
    Args:
        regularizer_name: String chosen from the available regularization choices.
    
    Returns:
        (fn, fn_deriv): A tuple with handles to the regularization function and its derivative.
    
    Raises:
        ValueError: If the specified regularizer is not available.
    """

    try:
        return _regularizers_with_deriv_dict[regularizer_name]
    except KeyError:
        raise ValueError('Specified regularizer not available.' \
                         f' Choose from - {list(_regularizers_with_deriv_dict)}')



def _L1_regularizer(x: npt.NDArray) -> npt.NDArray:
    return np.sum(np.abs(x))

def _L1_regularizer_deriv(x: npt.NDArray) -> npt.NDArray:
    return np.sign(x)


def _L2_regularizer(x: npt.NDArray) -> npt.NDArray:
    return np.sum(x ** 2)

def _L2_regularizer_deriv(x: npt.NDArray) -> npt.NDArray:
    return 2 * x


_regularizers_with_deriv_dict = {
    'L1': (_L1_regularizer, _L1_regularizer_deriv),
    'L2': (_L2_regularizer, _L2_regularizer_deriv),
}