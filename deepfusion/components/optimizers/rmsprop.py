from typing import override

import numpy.typing as npt

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from .optimizer import Optimizer


class RMSprop(Optimizer):
    """Root Mean Square propagation (RMSprop) optimizer that uses a moving average of the second
    moment of the loss derivative to perform updates.

    Attributes:
        data_shape: Shape of the Data object that the optimizer is associated with.
        beta: Hyperparameter that controls smoothing rate in the moving average of second moment.
        S: Exponentially weighted moving average of the second moment of the loss derivative.
        t: Counter for update steps. Used in bias correction of moving average.
        epsilon: Helps in maintaining numerical stability in case S becomes too small.
    """

    def __init__(self, data_shape: tuple[int, ...], beta: float = 0.9) -> None:
        super().__init__(data_shape)

        # Set the hyperparameter beta that controls the moving average of the second moment of deriv
        self.beta = beta
        # Initialize the weighted average of the second moment to 0
        self.S = np.zeros(data_shape)
        # Maintain counter of update steps for bias correction
        self.t = 0
        # Epsilon helps with numerical stability in case S becomes too small
        self.epsilon = 1e-8
    

    @override
    def step(self, data_deriv: npt.NDArray) -> npt.NDArray:
        self.t += 1
        self.S = self.beta * self.S + (1 - self.beta) * data_deriv ** 2
        S_corrected = self.S / (1 - self.beta ** self.t)
        return data_deriv / np.sqrt(S_corrected + self.epsilon)