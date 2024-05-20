from abc import ABC, abstractmethod

import numpy.typing as npt


class Optimizer(ABC):
    """Base class from which all optimizers should inherit.
    
    Attributes:
        data_shape: Shape of the Data object that the optimizer is associated with.
    """

    def __init__(self, data_shape: tuple[int, ...]) -> None:
        self.data_shape = data_shape
    
    
    @abstractmethod
    def step(self, data_deriv: npt.NDArray) -> npt.NDArray:
        """Returns the update tensor for the Data object given its derivative.
        
        Args:
            data_deriv: Derivative of the Data object at its current value for some loss function.
        
        Returns:
            np.ndarray: Returns a tensor to be used by the Data object for performing an update.
        """
        pass


    def __str__(self) -> str:
        print_optimizer = type(self).__name__
        return print_optimizer