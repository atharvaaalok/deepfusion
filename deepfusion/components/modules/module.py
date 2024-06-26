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
    
    available_modes = ['train', 'test']
    

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

        self.is_frozen = is_frozen if parameter_list is not None else True
        self.optimizer_details = optimizer_details if not is_frozen else None

        # Store the current module object in the outputs and input of the associated Data objects
        self.set_data_obj_in_out(inputs, output)

        
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
    

    def set_data_obj_in_out(self, inputs: list[Data], output: Data) -> None:
        """Stores the current module object in the outputs and input of associated Data objects.
        
        Args:
            inputs: List of Data objects whose values the module transforms to produce an output.
            output: Data object which stores the modules output after transforming the input values.
        """
        
        # Add the module to the input Data object's outputs list
        for input in inputs:
            input.outputs.append(self)
        
        # Store the module in the output Data object's input field
        output.input = self


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
    

    def __str__(self):
        """Returns a string representation of the Module object based on current attribute 
        values."""
        print_input = [ele.ID for ele in self.inputs]
        print_output = self.output.ID
        print_parameter = [param.ID for param in self.parameter_list]
        # Get attributes that may exist, if they don't set them to None
        print_mode = getattr(self, 'mode', None)
        print_is_regularized = getattr(self, 'is_regularized', None)
        print_regularizer_details = getattr(self, 'regularizer_details', None)

        print_module = (f'Module Object\n' + 40 * '-' + '\n' +
                        f'ID                      : {self.ID}\n' +
                        f'inputs                  : {print_input}\n' +
                        f'output                  : {print_output}\n' +
                        f'parameter_list          : {print_parameter}\n'
                        f'is_frozen               : {self.is_frozen}\n' +
                        f'optimizer_details       : {self.optimizer_details}\n' +
                        f'learning_rate           : {self.learning_rate}\n' +
                        f'different_at_train_test : {self.different_at_train_test}\n' +
                        f'mode                    : {print_mode}\n' +
                        f'is_regularizable        : {self.is_regularizable}\n' +
                        f'is_regularized          : {print_is_regularized}\n' +
                        f'regularizer_details     : {print_regularizer_details}\n')
        
        return print_module