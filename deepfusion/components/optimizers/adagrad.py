from .optimizer import Optimizer
from typing import override
import numpy.typing as npt

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()


class AdaGrad(Optimizer):
    """Adaptive Gradient (AdaGrad) optimizer that uses running sum of second moment of loss
    derivative to perform updates.

    Attributes:
        data_shape: Shape of the Data object that the optimizer is associated with.
        S: Running sum of the second moment of the loss derivative.
        epsilon: Helps in maintaining numerical stability in case S becomes too small.
    """

    def __init__(self, data_shape: tuple[int, ...]) -> None:
        super().__init__(data_shape)

        # Initialize the total sum of the second moment to 0
        self.S = np.zeros(data_shape)
        # Epsilon helps with numerical stability in case S becomes too small
        self.epsilon = 1e-8
    

    @override
    def step(self, data_deriv: npt.NDArray) -> npt.NDArray:
        self.S = self.S + data_deriv ** 2
        return data_deriv / np.sqrt(self.S + self.epsilon)