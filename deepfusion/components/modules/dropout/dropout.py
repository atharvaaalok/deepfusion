from typing import override

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from ..module import Module
from ...data.data import Data


class Dropout(Module):
    """Dropout module that drops some neurons at random during forward pass in training.

    Attributes:
        ID:
            A unique string identifier for the Module object.
        inputs:
            List of Data objects whose values the module transforms to produce an output.
        output:
            Data object which stores the module's output after transforming the input values.
        parameter_list:
            Not used by the module. Set to default value by the Module base class.
        learning_rate:
            Not used by the module. Set to default value by the Module base class.
        is_frozen:
            Not used by the module. Set to default value by the Module base class.
        optimizer_details:
            Not used by the module. Set to default value by the Module base class.
        mode:
            Module works differently in train and test. Maintain state to use appropriate values.
        p_keep:
            Probability of keeping a particular neuron during forward pass in training.
        cache:
            Dictionary that stores values during forward pass that will be used in backward pass.
    """

    different_at_train_test = True
    is_regularizable = False


    def __init__(self, ID: str, inputs: list[Data], output: Data, p_keep: float = 0.5) -> None:
        """Initialize the Dropout module based on ID, inputs, output and optional parameters."""

        # Go through checks first
        assert inputs[0].shape == output.shape, 'For Dropout input and output shape should be same.'

        super().__init__(ID, inputs, output)

        # Works differently in train and test mode. Maintain state to use appropriate values
        self.mode = 'train'
        # Probability of keeping a particular neuron during forward pass in training
        self.p_keep = p_keep
        # Maintain cache for mask used in forward pass during training
        self.cache = {'mask': 0}
    

    @override
    def forward(self) -> None:
        if self.mode == 'train':
            # Drop neurons at randon based on probability p_keep
            mask = np.random.rand(*self.inputs[0].val.shape) < self.p_keep
            # Apply the mask
            self.output.val = self.inputs[0].val * mask
            # Cache the mask to use it during the backward pass
            self.cache['mask'] = mask

        elif self.mode == 'test':
            # Don't drop neurons, use all of them, but scale down values by probability p_keep
            self.output.val = self.inputs[0].val * self.p_keep


    @override
    def backward(self) -> None:
        self.inputs[0].deriv += self.output.deriv * self.cache['mask']
    

    def set_mode(self, mode: str) -> None:
        """Sets the mode (training/testing) for the module.
        
        Args:
            mode: String determining whether the network is in 'train' or 'test' mode.
        
        Raises:
            ValueError: If the specified mode is not available.
        """
        if mode not in self.available_modes:
            raise ValueError(f'Specified mode is not available. Choose from {self.available_modes}')
        
        self.mode = mode