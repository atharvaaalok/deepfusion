from typing_extensions import override

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from ..module import Module
from ...data.data import Data
from ...optimizers import DEFAULT_OPTIMIZER_DETAILS


class LayerNorm(Module):
    """Layer Normalization module that given an input X, normalizes it across features of each
    training example separately and scales and shifts the result with learned parameters.

    Layer normalization takes an input X and first normalizes it to produce X_hat = (X - Mean) / Var
    , where the mean and variance are calculated across the features for each training example in
    the mini-batch separately. The final output is produced by scaling and shifting X_hat as
    (Out = gamma * X_hat + beta), where gamma and beta are learnable parameters of the same
    dimension as input features.

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
        gamma:
            Scaling parameter in (Out = gamma * X_hat + beta) that is learnable.
        beta:
            Shift parameter in (Out = gamma * X_hat + beta) that is learnable.
        epsilon:
            Helps in maintaining numerical stability in case variance becomes too small.
        cache:
            Dictionary that stores values during forward pass that will be used in backward pass.
    """

    different_at_train_test = False
    is_regularizable = False


    def __init__(
        self,
        ID: str,
        inputs: list[Data],
        output: Data,
        learning_rate: float = 1e-6,
        is_frozen: bool = False,
        optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS,
    ) -> None:
        """Initializes Layer Normalization module based on ID, inputs, output and other optional
        parameters."""

        # Go through checks first
        assert inputs[0].shape == output.shape, \
            'For Layer Norm input and output shape should be same.'
        
        # Determine whether input is a vector or convolutional volume
        if len(inputs[0].shape) == 3:
            conv_input = True
        else:
            conv_input = False
        
        # Define the parameters gamma and beta associated with the module
        gamma_shape = (1, output.shape[0], 1, 1) if conv_input else output.shape
        beta_shape = (1, output.shape[0], 1, 1) if conv_input else output.shape
        self.gamma = Data(ID = ID + '_gamma',
                          shape = gamma_shape,
                          val = np.ones(gamma_shape),
                          is_frozen = is_frozen,
                          optimizer_details = optimizer_details)
        
        self.beta = Data(ID = ID + '_beta',
                         shape = (beta_shape),
                         val = np.zeros(beta_shape),
                         is_frozen = is_frozen,
                         optimizer_details = optimizer_details)

        parameter_list = [self.gamma, self.beta]

        super().__init__(ID, inputs, output, parameter_list = parameter_list,
                         learning_rate = learning_rate, is_frozen = is_frozen,
                         optimizer_details = optimizer_details)

        # Epsilon helps with numerical stability in case variance becomes too small
        self.epsilon = 1e-8

        # Initialize cache which will store values used in forward pass that are needed for backprop
        self.cache = {'conv_input': conv_input, 'mu': 0, 'sigma_sq': 0}
    

    @override
    def forward(self) -> None:
        X = self.inputs[0].val

        # Calculate mean and variance across features for each training example
        axes_to_norm = (1, 2, 3) if self.cache['conv_input'] else 1
        mu = np.mean(X, axis = axes_to_norm, keepdims = True)
        sigma_sq = np.var(X, axis = axes_to_norm, keepdims = True)

        # Cache mu and sigma_sq
        self.cache['mu'] = mu
        self.cache['sigma_sq'] = sigma_sq
        # Not saving X_hat to cache as it will take up a lot more memory

        # Normalize X
        X_hat = (X - mu) / np.sqrt(sigma_sq + self.epsilon)

        # Scale and shift X_hat to produce the output
        self.output.val = self.gamma.val * X_hat + self.beta.val
    

    @override
    def backward(self) -> None:
        out_deriv = self.output.deriv

        # Retrieve mu and sigma_sq from cache
        mu = self.cache['mu']
        sigma_sq = self.cache['sigma_sq']
        # Recompute X_hat
        X = self.inputs[0].val
        X_hat = (X - mu) / np.sqrt(sigma_sq + self.epsilon)

        # Set derivatives for the parameters beta and gamma
        axes_to_sum = (0, 2, 3) if self.cache['conv_input'] else 0
        self.beta.deriv += np.sum(out_deriv, axis = axes_to_sum, keepdims = True)
        self.gamma.deriv += np.sum(out_deriv * X_hat, axis = axes_to_sum, keepdims = True)

        # Calculate derivative of loss w.r.t. X_hat
        dX_hat = self.gamma.val * out_deriv

        # Compute derivative of loss w.r.t. input X, which is sum of 3 parts
        axes_to_norm = (1, 2, 3) if self.cache['conv_input'] else 1
        dX_1 = dX_hat
        dX_2 = - np.mean(dX_hat, axis = axes_to_norm, keepdims = True)
        dX_3 = - X_hat * np.mean(X_hat * dX_hat, axis = axes_to_norm, keepdims = True)

        self.inputs[0].deriv += (1 / np.sqrt(sigma_sq + self.epsilon)) * (dX_1 + dX_2 + dX_3)