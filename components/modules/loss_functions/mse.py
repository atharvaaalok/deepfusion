import numpy as np

from ..module import Module


class MSE(Module):

    different_at_train_test = False
    is_regularizable = False


    def __init__(self, ID, inputs, output):
        super().__init__(ID, inputs, output)
    
    
    def forward(self):
        batch_size = self.inputs[0].val.shape[1]
        self.output.val = (1 / batch_size) * (1 / 2) * np.sum((self.inputs[0].val - self.inputs[1].val) ** 2)

        self.output.deriv = 1.0


    def backward(self):
        batch_size = self.inputs[0].val.shape[1]

        self.inputs[0].deriv = (1 / batch_size) * (self.inputs[0].val - self.inputs[1].val) * self.output.deriv
        self.inputs[1].deriv = (1 / batch_size) * (self.inputs[1].val - self.inputs[0].val) * self.output.deriv