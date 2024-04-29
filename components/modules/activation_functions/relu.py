import numpy as np

from ..module import Module


class Relu(Module):

    different_at_train_test = False
    is_regularizable = False


    def __init__(self, ID, inputs, output):
        super().__init__(ID, inputs, output)
    

    def forward(self):
        self.output.val = np.maximum(self.inputs[0].val, 0.0)
    

    def backward(self):
        self.inputs[0].deriv = self.output.deriv * np.where(self.inputs[0].val >= 0.0, 1.0, 0.0)