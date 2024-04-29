from abc import ABC, abstractmethod


class Optimizer:

    def __init__(self, data_shape):
        self.data_shape = data_shape
    
    @abstractmethod
    def step(self, data_deriv):
        pass