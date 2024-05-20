import numpy.typing as npt
import matplotlib.pyplot as plt

from deepfusion.utils.backend import Backend
np = Backend.get_array_module()
from deepfusion.components.data import Data
from deepfusion.components.modules import Module
from deepfusion.components.net import Net
from deepfusion.utils.colors import red, cyan, color_end


def automate_training(
    net: Net,
    X_train: npt.NDArray,
    Y_train: npt.NDArray,
    X_val: npt.NDArray,
    Y_val: npt.NDArray,
    B: int = 64,
    epochs: int = 1000,
    print_cost_every: int = 100,
    learning_rate: float = 0.001,
    lr_decay: float = 0.01,
) -> None:
    """Automates the learning procedure for neural networks with a single input and loss."""

    # Get input, target label and loss node
    x = net.topological_order[0]
    for node in reversed(net.topological_order):
        if isinstance(node, Data):
            if node.input is None:
                # Set target node
                y = node
                break
    loss = net.topological_order[-1]

    # Set initial learning rate
    net.set_learning_rate(learning_rate)

    # Initialize lists for storing training and validation errors
    epoch_list = []
    cost_train = []
    cost_val = []

    # Initialize interactive plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    train_line, = ax.plot([], [], label = 'Train Cost', color = '#277cb3', linewidth = 2)
    val_line, = ax.plot([], [], label = 'Val Cost', color = '#fa7f13', linewidth = 2)
    ax.legend()

    # Train the network
    for epoch in range(1, epochs + 1):
        # Turn on training mode
        net.set_mode('train')

        # Generate mini batch of training examples
        idx = np.random.choice(X_train.shape[0], size = B, replace = False)
        x.val = X_train[idx, :]
        y.val = Y_train[idx, :]

        # Run the forward pass
        net.forward()            

        # Run the backward pass
        net.backward()

        # Update the parameters using the gradients
        net.update()

        # Clear gradients after the parameters have been updated so that they don't accumulate
        net.clear_grads()

        # Implement decay in the learning rate
        net.set_learning_rate(learning_rate = learning_rate / (1 + lr_decay * epoch))


        # Print cost every few steps
        if epoch % print_cost_every == 0 or epoch == 1:
            J_train = loss.val
            num_digits = len(str(epochs))
            epoch_list.append(epoch)
            cost_train.append(J_train)

            # Evaluate current model on mini-batch of validation data
            net.set_mode('test')
            x.val = X_val
            y.val = Y_val
            net.forward()
            J_val = loss.val
            cost_val.append(J_val)
            
            print(f'{red}Epoch:{color_end} [{epoch:{num_digits}}/{epochs}].  ' \
                    f'{cyan}Train Cost:{color_end} {J_train:11.6f}.  ' \
                    f'{cyan}Val Cost:{color_end} {J_val:11.6f}')
            
            # Update the plot
            plt.ion()
            train_line.set_xdata(epoch_list)
            train_line.set_ydata(cost_train)
            val_line.set_xdata(epoch_list)
            val_line.set_ydata(cost_val)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

    plt.ioff()
    plt.show()