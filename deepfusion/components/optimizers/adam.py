from typing import override

import numpy.typing as npt

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from .optimizer import Optimizer


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer that uses a moving average of the first and
    second moment of the loss derivative to perform updates.

    Attributes:
        data_shape: Shape of the Data object that the optimizer is associated with.
        beta1: Hyperparameter that controls smoothing rate in the moving average of first moment.
        beta2: Hyperparameter that controls smoothing rate in the moving average of second moment.
        V: Exponentially weighted moving average of the first moment of the loss derivative.
        S: Exponentially weighted moving average of the second moment of the loss derivative.
        t: Counter for update steps. Used in bias correction of moving average.
        epsilon: Helps in maintaining numerical stability in case S becomes too small.
    """

    def __init__(self, data_shape: tuple[int, ...], beta1: float = 0.9, beta2: float = 0.999) -> None:
        super().__init__(data_shape)

        # Set the hyperparameters beta1 and beta2 that control the moving average of the first and
        # second moment of deriv
        self.beta1 = beta1
        self.beta2 = beta2
        # Initialize the weighted average of the first and second moment to 0
        self.V = np.zeros(data_shape)
        self.S = np.zeros(data_shape)
        # Maintain counter of update steps for bias correction
        self.t = 0
        # Epsilon helps with numerical stability in case S becomes too small
        self.epsilon = 1e-8


    @override
    def step(self, data_deriv: npt.NDArray) -> npt.NDArray:
        self.t += 1

        self.V = self.beta1 * self.V + (1 - self.beta1) * data_deriv
        self.S = self.beta2 * self.S + (1 - self.beta2) * data_deriv ** 2

        V_corrected = self.V / (1 - self.beta1 ** self.t)
        S_corrected = self.S / (1 - self.beta2 ** self.t)

        return V_corrected / np.sqrt(S_corrected + self.epsilon)