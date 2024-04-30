from .optimizer import Optimizer
from typing import override
import numpy.typing as npt


class SGD(Optimizer):

    def __init__(self, data_shape: tuple[int, ...]) -> None:
        super().__init__(data_shape)
    

    @override
    def step(self, data_deriv: npt.NDArray) -> npt.NDArray:
        return data_deriv