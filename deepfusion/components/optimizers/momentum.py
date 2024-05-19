from .optimizer import Optimizer
from typing import override
import numpy.typing as npt

from ...utils.backend import Backend
np = Backend.get_array_module()


class Momentum(Optimizer):
    """Momentum optimizer that uses a moving average of the first moment of the loss derivative to
    perform updates.

    Attributes:
        data_shape: Shape of the Data object that the optimizer is associated with.
        beta: Hyperparameter that controls the smoothing rate in the moving average of first moment.
        V: Exponentially weighted moving average of the first moment of the loss derivative.
        t: Counter for update steps. Used in bias correction of moving average.
    """

    def __init__(self, data_shape: tuple[int, ...], beta: float = 0.9) -> None:
        super().__init__(data_shape)

        # Set the hyperparameter beta that controls the moving average for the velocity
        self.beta = beta
        # Initialize the velocity parameter to 0
        self.V = np.zeros(data_shape)
        # Maintain counter of update steps for bias correction
        self.t = 0


    @override
    def step(self, data_deriv: npt.NDArray) -> npt.NDArray:
        self.t += 1
        self.V = self.beta * self.V + (1 - self.beta) * data_deriv
        V_corrected = self.V / (1 - self.beta ** self.t)
        return V_corrected