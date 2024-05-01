from abc import ABC, abstractmethod
from typing import Optional

from ..data.data import Data
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS


class Module(ABC):
    """Base class from which all modules should inherit.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the modules output after transforming the input values.
        parameter_list:
            List of parameter Data objects associated with the module. Empty if module has no
            parameters.
        learning_rate:
            Controls the step size of the update for the module's parameters.
        is_frozen:
            Boolean that decides if updates will be made to module's parameters or not.
        optimizer_details:
            Dictionary containing name of the optimizer and a dictionary of its associated
            hyperparameters.
    """

    @property
    @abstractmethod
    def different_at_train_test():
        pass


    @property
    @abstractmethod
    def is_regularizable():
        pass


    def __init__(
        self,
        ID: str,
        inputs: list[Data],
        output: Data,
        parameter_list: Optional[list[Data]] = None,
        learning_rate: float = 1e-6,
        is_frozen: bool = False,
        optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS,
    ) -> None:
        """Initializes Module object based on ID, inputs, output and other optional parameters.

        Raises:
            ValueError: If input is not a list of Data objects or the output is not a Data object.
        """

        self.ID = ID

        self._check_input_output_type(inputs, output)
        self.inputs = inputs
        self.output = output

        self.parameter_list = parameter_list if parameter_list is not None else []

        self.learning_rate = learning_rate
        self.set_learning_rate(learning_rate)

        self.is_frozen = is_frozen
        self.optimizer_details = optimizer_details if not is_frozen else None

        
    @staticmethod
    def _check_input_output_type(inputs: list[Data], output: Data) -> None:
        """Utility function that checks if the inputs and the output provided are Data objects.

        Args:
            inputs: List of Data objects.
            output: Data object.

        Raises:
            ValueError: If input is not a list of Data objects or the output is not a Data object.
        """
        nodes = inputs + [output]

        for node in nodes:
            if not isinstance(node, Data):
                raise ValueError('Inputs and Output provided should be objects of the Data class.')


    @abstractmethod
    def forward(self) -> None:
        """Sets the value attribute for output Data object by transforming the input Data objects'
        value attributes."""
        pass


    @abstractmethod
    def backward(self) -> None:
        """Sets the deriv attribute of the input Data objects by appropriately transforming the
        output Data object's deriv attribute."""
        pass


    def update(self) -> None:
        """Updates each parameter Data object's value if module is not frozen."""
        if not self.is_frozen:
            for parameter in self.parameter_list:
                parameter.update()
    

    def freeze(self) -> None:
        """Freeze each parameter Data object so that updates to its value will no longer be made."""
        self.is_frozen = True
        self.optimizer_details = None

        for parameter in self.parameter_list:
            parameter.freeze()
    

    def unfreeze(self, optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS) -> None:
        """Unfreeze each parameter Data object to allow updates to be made to their values."""
        self.is_frozen = False
        self.optimizer_details = optimizer_details

        for parameter in self.parameter_list:
            parameter.unfreeze(optimizer_details)
    

    def clear_grads(self) -> None:
        """Set deriv attribute for each parameter object to 0."""
        for parameter in self.parameter_list:
            parameter.clear_grads()
    

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate for each parameter Data object to the specified value."""
        self.learning_rate = learning_rate
        for parameter in self.parameter_list:
            parameter.set_learning_rate(learning_rate)
    

    def set_optimizer(self, optimizer_details: dict) -> None:
        """Set optimizer (if module not frozen) for each parameter Data object.

        Args:
            optimizer_details: Dictionary containing name of the optimizer and a dictionary of its
            associated hyperparameters.
        """
        
        if not self.is_frozen:
            self.optimizer_details = optimizer_details
            for parameter in self.parameter_list:
                parameter.set_optimizer(optimizer_details)