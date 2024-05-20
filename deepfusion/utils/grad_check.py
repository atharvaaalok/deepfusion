from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from deepfusion.components.data import Data
from deepfusion.components.net import Net


def gradient_checker(net: Net, data_obj: Data, h: float = 1e-6) -> None:
    """Performs comparison between analytic and numeric gradient for a data object in a neural net.
    
    Args:
        net: Neural network for which the gradient checking is to be done.
        data_obj: Data object w.r.t. whose value the gradients are to be calculated.
        loss_obj: Loss object whose value is the output of the neural network.
        h: The step size used in calculating the value of the gradient.
    """

    # Clear any residual gradients before performing any operations
    net.clear_grads()

    # Determine if the data_obj is a data node or is a parameter inside a module
    if data_obj in net.topological_order:
        data_node = True
    else:
        data_node = False
    
    # Run the neural network forward once to set the data object values
    net.forward()
    
    ## Calculate analytic gradient
    # Set random seed before running forward. Helps if dropout layers are present
    np.random.seed(0)
    # Run forward() if a parameter is passed, else if a data_node is passed run forward_from_node()
    net.forward_from_node(data_obj) if data_node else net.forward()
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

        # Add up gradient contributions due to all loss functions present in the network
        for loss_obj in net.root_nodes:
            # Evaluate network at x + h
            x[idx] = old_x + h
            np.random.seed(0)
            net.forward_from_node(data_obj) if data_node else net.forward()
            f_plus = loss_obj.val

            # Evaluate network at x - h
            x[idx] = old_x - h
            np.random.seed(0)
            net.forward_from_node(data_obj) if data_node else net.forward()
            f_minus = loss_obj.val

            # Compute numeric gradient at the current x index
            numeric_grad[idx] += (f_plus - f_minus) / (2 * h)

        # Restore old value of x
        x[idx] = old_x

        # Get the next index at which to calculate numeric gradient
        iter_x.iternext()
    
    # Calculate relative error for each value of x, take care of corner case when both are 0
    idx = (analytic_grad == 0) & (numeric_grad == 0)
    rel_error = np.ones(numeric_grad.shape)
    rel_error[idx] = 0
    epsilon = 1e-8
    rel_error[~idx] = (np.abs(analytic_grad[~idx] - numeric_grad[~idx]) /
                          np.maximum(epsilon, np.abs(analytic_grad[~idx]) + np.abs(numeric_grad[~idx])))
    
    # Print gradients and errors
    print('\nAnalytic gradient\n', analytic_grad)
    print('\nNumeric gradient\n', numeric_grad)
    print('\nRelative error\n', rel_error)
    print('\nMax error\n', np.max(rel_error))