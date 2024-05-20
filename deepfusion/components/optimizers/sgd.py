from .optimizer import Optimizer
from typing import override
import numpy.typing as npt


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer that performs update with plain loss derivative.
    
    Attributes:
        data_shape: Shape of the Data object that the optimizer is associated with.
    """

    def __init__(self, data_shape: tuple[int, ...]) -> None:
        super().__init__(data_shape)
    

    @override
    def step(self, data_deriv: npt.NDArray) -> npt.NDArray:
        return data_deriv