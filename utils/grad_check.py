import numpy as np

from ..components.net import Net
from ..components.data import Data


def gradient_checker(net: Net, data_obj: Data, loss_obj: Data, h: float = 1e-5) -> None:
    """Performs comparison between analytic and numeric gradient for a data object in a neural net.
    
    Args:
        net: Neural network for which the gradient checking is to be done.
        data_obj: Data object w.r.t. whose value the gradients are to be calculated.
        loss_obj: Loss object whose value is the output of the neural network.
        h: The step size used in calculating the value of the gradient.
    """
    
    ## Calculate analytic gradient
    net.forward()
    net.backward()
    analytic_grad = data_obj.deriv


    ## Calculate numeric gradient

    x = data_obj.val
    numeric_grad = np.zeros(x.shape)

    # Create an iterator over the x values
    iter_x = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])

    while not iter_x.finished:

        # Get the index to change and find numeric gradient
        idx = iter_x.multi_index
        old_x = x[idx]

        # Evaluate network at x + h
        x[idx] = old_x + h
        net.forward()
        f_plus = loss_obj.val

        # Evaluate network at x - h
        x[idx] = old_x - h
        net.forward()
        f_minus = loss_obj.val

        # Compute numeric gradient at the current x index
        numeric_grad[idx] = (f_plus - f_minus) / (2 * h)

        # Restore old value of x
        x[idx] = old_x

        # Get the next index at which to calculate numeric gradient
        iter_x.iternext()
    
    # Calculate relative error for each value of x
    rel_error = (np.abs(analytic_grad - numeric_grad) /
                 np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)))
    
    # Print gradients and errors
    print('\nAnalytic gradient\n', analytic_grad)
    print('\nNumeric gradient\n', numeric_grad)
    print('\nRelative error\n', rel_error)
    print('\nMax error\n', np.max(rel_error))