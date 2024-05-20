from deepfusion.utils.backend import Backend
np = Backend.get_array_module()

import numpy.typing as npt
from typing import Optional

from .regularizers import Regularizer
from ..optimizers import get_optimizer, DEFAULT_OPTIMIZER_DETAILS


class Data:
    """Data objects hold tensors and derivatives w.r.t. the tensor value of some loss function.

    Data objects can be used as constants or as parameters based on whether they are frozen or
    unfrozen respectively. They get updates using their defined optimizer objects and can also be
    regularized using a regularizer. Each data object has its own learning rate which controls the
    update step size.

    Attributes:
        ID:
            A unique string identifier for the Data object.
        shape:
            This is the dimension of the tensor that the object holds. The shape always has one
            dimension higher (with extra last dimension set to 1, as if mini-batch = 1) than the
            original dimensionality of the data to account for mini-batches. Eg: a 1D vector
            [1, 2, 3] (original shape = (3, )) should be passed as a 2D matrix with last dimension 1
            i.e. [[1], [2], [3]] (shape = (3, 1)).
        val:
            Data held by the object. The last dimension defines the mini-batch size.
        deriv:
            Holds the derivative of some loss function at the current data value.
        is_frozen:
            Boolean that decides if updates will be made to the value or not.
        optimizer:
            Optimizer object for the data object.
        learning_rate:
            Controls the step size of the update.
        is_regularized:
            Boolean that decides if the data contributes to regularization loss.
        regularizer:
            Regularizer object that holds the regularization strength and function.
        input:
            Stores Module object the output of which is stored in the Data object.
        outputs:
            List of Module objects which use the Data object's value.
    """

    def __init__(
        self,
        ID: str,
        shape: tuple[int, ...],
        val: npt.NDArray = None,
        is_frozen: bool = True,
        optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS,
        learning_rate: float = 1e-6,
        is_regularized: bool = False,
        regularizer_details: Optional[dict] = None,
    ) -> None:
        """Initializes Data instance based on ID, shape and other optional parameters.
        
        Raises:
            ValueError: If data is regularized but regularizer details are not provided.
        """
        
        self.ID = ID

        self.shape = shape
        self.val = np.zeros(shape) if val is None else val
        self.deriv = np.zeros(shape)

        self.is_frozen = is_frozen

        self.optimizer = get_optimizer(shape, optimizer_details) if not is_frozen else None

        self.learning_rate = learning_rate

        self.is_regularized = is_regularized
        if is_regularized and regularizer_details is None:
            raise ValueError('If Data is regularized, regularizer details must be provided.')
        self.regularizer = None if regularizer_details is None else Regularizer(regularizer_details)

        # Create empty lists for storing input and outputs. These will be filled by modules.
        self.input = None
        self.outputs = []
    
    
    def update(self) -> None:
        """Updates the value (if not frozen) using steps from optimizer and regularizer."""
        if not self.is_frozen:
            optimizer_step = self.optimizer.step(self.deriv)
            if self.is_regularized:
                reg_step = self.regularizer.reg_strength * self.regularizer.reg_fn_deriv(self.val)
            else:
                reg_step = 0.0
            self.val -= self.learning_rate * (optimizer_step + reg_step)

        # Clear gradients after updating the daja object
        self.clear_grads()


    def freeze(self) -> None:
        """Freeze the data object so that updates to its value will no longer be made."""
        self.is_frozen = True
        self.optimizer = None


    def unfreeze(self, optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS) -> None:
        """Unfreeze the data object to allow updates to be made to its value.
        
        Args:
            optimizer_details: Dictionary of optimizer name and hyperparameters to use for updates.
        """
        self.is_frozen = False
        self.optimizer = get_optimizer(self.shape, optimizer_details)


    def clear_grads(self) -> None:
        """Set deriv attribute to scalar 0."""
        self.deriv *= 0


    def set_regularization(self, regularizer_details: dict) -> None:
        """Regularize the data using specified regularization strength and function.
        
        Args:
            regularizer_details: Dictionary with regularizer strength and function name.
        """
        self.is_regularized = True
        self.regularizer = Regularizer(regularizer_details)


    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    

    def set_optimizer(self, optimizer_details: dict) -> None:
        """Set an optimizer for the Data object that can be used to update the value.

        Args:
            optimizer_details: Dictionary containing name of the optimizer and a dictionary of its
            associated hyperparameters.
        """
        if not self.is_frozen:
            self.optimizer = get_optimizer(self.shape, optimizer_details)