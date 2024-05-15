import numpy as np
import numpy.typing as npt


# Sigmoid function and derivative
def _sigmoid(x: npt.NDArray) -> npt.NDArray:
    """Sigmoid function that given an input X computes 1 / (1 + exp(-X)) element-wise.

    Calculations are done differently for positive and negative x values to prevent overflow
    situations. This is because for large negative X values exp(-X) is a very large positive
    value and overflow warning is thrown.
    
    Args:
        x: Input tensor for which the function is to be applied element-wise.
    
    Returns:
        npt.NDArray: Tensor of the same shape as x, containing the sigmoid-transformed values.
    """

    def _positive_sigmoid(x):
        """Utility function that calculates sigmoid for positive x values."""
        return 1 / (1 + np.exp(-x))
    
    def _negative_sigmoid(x):
        """Utility function that calculates sigmoid for negative x values."""
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)
    
    # Calculate sigmoid separately for positive and negative x values
    # Interesting details can be found in this stackoverflow answer:
    # https://stackoverflow.com/a/64717799/12842649
    pos_x_mask = x >= 0
    neg_x_mask = ~pos_x_mask

    sigmoid_x = np.zeros(x.shape)
    sigmoid_x[pos_x_mask] = _positive_sigmoid(x[pos_x_mask])
    sigmoid_x[neg_x_mask] = _negative_sigmoid(x[neg_x_mask])

    return sigmoid_x


def _sigmoid_deriv(x: npt.NDArray) -> npt.NDArray:
    """Derivative of the sigmoid function.
    
    Args:
        x: Input tensor at which the derivative of the sigmoid function is to be calculated.
    
    Returns:
        npt.NDArray: Tensor of the same shape as x, containing the sigmoid derivative values.
    """

    sigmoid_x = _sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)



# Tanh function and derivative
def _tanh(x: npt.NDArray) -> npt.NDArray:
    """Tanh function that given an input X computes (exp(2x) - 1) / (exp(2x) + 1) element-wise.

    Calculations are done differently for positive and negative x values to prevent overflow
    situations. This is because for large positive X values exp(X) is a very large positive value
    and overflow warning is thrown.
    
    Args:
        x: Input tensor for which the function is to be applied element-wise.
    
    Returns:
        npt.NDArray: Tensor of the same shape as x, containing the tanh-transformed values.
    """

    def _positive_tanh(x):
        """Utility function that calculates tanh for positive x values."""
        exp_neg_2x = np.exp(-2 * x)
        return (1 - exp_neg_2x) / (1 + exp_neg_2x)
    
    def _negative_tanh(x):
        """Utility function that calculates tanh for negative x values."""
        exp_2x = np.exp(2 * x)
        return (exp_2x - 1) / (exp_2x + 1)

    # Calculate tanh separately for positive and negative x values
    # Logic for splitting as such is similar to that explained for the sigmoid function
    # Interesting details (for sigmoid) can be found in this stackoverflow answer:
    # https://stackoverflow.com/a/64717799/12842649
    pos_x_mask = x >= 0
    neg_x_mask = ~pos_x_mask

    tanh_x = np.zeros(x.shape)
    tanh_x[pos_x_mask] = _positive_tanh(x[pos_x_mask])
    tanh_x[neg_x_mask] = _negative_tanh(x[neg_x_mask])

    return tanh_x


def _tanh_deriv(x: npt.NDArray) -> npt.NDArray:
    """Derivative of the tanh function.
    
    Args:
        x: Input tensor at which the derivative of the tanh function is to be calculated.
    
    Returns:
        npt.NDArray: Tensor of the same shape as x, containing the tanh derivative values.
    """

    tanh_x = _tanh(x)
    return 1 - tanh_x ** 2



# Softmax function
def _softmax(x: npt.NDArray) -> npt.NDArray:
    """Softmax function that given an input vector X computes y_ij = exp(x_ij) / (sum_k(exp(x_ik))).
    
    Args:
        x: Input tensor for which the function is to be applied element-wise.
    
    Returns:
        npt.NDArray: Tensor of the same shape as x, containing the softmax-transformed values.
    """

    # Subtract the maximum value of each column in x to prevent overflow when calculating exp(x)
    x = x - np.max(x, axis = 1, keepdims = True)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis = 1, keepdims = True)

    return softmax_x