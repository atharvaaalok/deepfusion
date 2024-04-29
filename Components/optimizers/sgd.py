from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, data_shape):
        super().__init__(data_shape)
    
    def update(self, data_deriv):
        return data_deriv