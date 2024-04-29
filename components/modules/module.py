from abc import ABC, abstractmethod

from ..data.data import Data
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS

class Module(ABC):

    @property
    @abstractmethod
    def different_at_train_test():
        pass


    @property
    @abstractmethod
    def is_regularizable():
        pass


    def __init__(self, ID, inputs, output, parameter_list = None, learning_rate = 1e-6, is_frozen = False, optimizer_details = DEFAULT_OPTIMIZER_DETAILS):
        self.ID = ID

        self.check_input_output_type(inputs, output)
        self.inputs = inputs
        self.output = output

        self.parameter_list = parameter_list if parameter_list is not None else []

        self.learning_rate = learning_rate
        self.set_learning_rate(learning_rate)

        self.is_frozen = is_frozen
        self.optimizer_details = optimizer_details if not is_frozen else None

        
    @staticmethod
    def check_input_output_type(inputs, output):
        nodes = inputs + [output]

        for node in nodes:
            if not isinstance(node, Data):
                raise ValueError('Inputs and Output provided should be objects of the Data class.')


    @abstractmethod
    def forward(self):
        pass


    @abstractmethod
    def backward(self):
        pass


    def update(self):
        if not self.is_frozen:
            for parameter in self.parameter_list:
                parameter.update()
    

    def freeze(self):
        self.is_frozen = True
        self.optimizer_details = None

        for parameter in self.parameter_list:
            parameter.freeze()
    

    def unfreeze(self, optimizer_details = DEFAULT_OPTIMIZER_DETAILS):
        self.is_frozen = False
        self.optimizer_details = optimizer_details

        for parameter in self.parameter_list:
            parameter.unfreeze(optimizer_details)
    

    def clear_grads(self):
        for parameter in self.parameter_list:
            parameter.clear_grads()
    

    def set_learning_rate(self, learning_rate):
        for parameter in self.parameter_list:
            parameter.set_learning_rate(learning_rate)