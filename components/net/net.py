from typing import Optional
import numpy.typing as npt
import time
from graphviz import Source
import matplotlib.pyplot as plt
import pickle
from ...utils.backend import Backend
np = Backend.get_array_module()

from ..data.data import Data
from ..modules.module import Module
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS
from ...utils.colors import red, cyan, color_end


class Net:
    """Directed Acyclic Graph that is composed of alternating data and module nodes. Neural network
    training happens through this net object.

    Attributes:
        ID:
            A unique string identifier for the Net object.
        root_nodes:
            List of leaf nodes which represent losses.
        learning_rate:
            Controls the step size of the update for the network's parameters.
        is_frozen:
            Boolean that decides if updates will be made to network's parameters or not.
        optimizer_details:
            Dictionary containing name of the optimizer and a dictionary of its associated
            hyperparameters.
        is_regularized:
            Boolean that decides if some parameters for the network are regularized.
        regularizer_details:
            Dictionary containing the regularization strength and the regularizer name.
        mode:
            Network works differently in train and test. Maintain state to use appropriate values.
        graph:
            A dictionary of node objects that represents the directed acyclic graph.
        graph_visual:
            A dictionary of node object IDs that represents the directed acyclic graph in human
            readable format.
        topological_order:
            A list of the nodes in the network in a topologically sorted order. Order:
            Inputs -> Losses.
        node_lookup:
            A dictionary mapping node IDs to node objects. Helpful for retrieving a node based on
            its ID.
    """

    available_modes = ['train', 'test']
    

    def __init__(
        self,
        ID: str,
        root_nodes: list[Data],
        learning_rate: float = 1e-6,
        optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS,
        is_regularized: bool = False,
        regularizer_details: Optional[dict] = None,
    ) -> None:
        """Initializes a Net based on ID, list of root nodes and other optional parameters."""
        
        self.ID = ID
        
        self.root_nodes = root_nodes

        self.graph = {}
        self.graph_visual = {}
        self.topological_order = []
        self.node_lookup = {}

        self.learning_rate = learning_rate

        self.is_frozen = False
        self.optimizer_details = optimizer_details if not self.is_frozen else None

        self.is_regularized = is_regularized
        self.regularizer_details = regularizer_details if is_regularized else None

        self.mode = 'train'
        
        self.run_setup()
    

    def run_setup(self) -> None:
        """Run by __init__() on network initialization to set topological order, graph dictionary
        and node lookup attributes. Also sets optimizer and regularization details."""
        self.topological_sort()
        self.create_graph_dicts()
        self.create_lookup()

        self.set_optimizer(self.optimizer_details)

        if self.is_regularized:
            self.set_regularization(self.regularizer_details)


    def topological_sort(self) -> None:
        """Creates a topological sort for the network based on the root nodes of the net. Order of
        nodes: Inputs -> Losses."""
        topological_order = []
        visited_nodes = set()

        def _topological_sort_util(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                if isinstance(node, Data):
                    if node.input is not None:
                        _topological_sort_util(node.input)
                elif isinstance(node, Module):
                    for child_node in node.inputs:
                        _topological_sort_util(child_node)
                
                topological_order.append(node)
        
        for root_node in self.root_nodes:
            _topological_sort_util(root_node)
        
        self.topological_order = topological_order
    

    def create_graph_dicts(self) -> None:
        """Sets the graph and graph_visual attributes for the network. Requires the topological sort
        to be already available."""
        for node in self.topological_order:
            if isinstance(node, Data):
                self.graph[node] = node.outputs
                self.graph_visual[node.ID] = [child.ID for child in node.outputs]
            elif isinstance(node, Module):
                self.graph[node] = [node.output]
                self.graph_visual[node.ID] = [node.output.ID]


    def create_lookup(self) -> None:
        """Sets the node_lookup attribute for the network. Requires the topological sort to be
        already available."""
        for node in self.topological_order:
            self.node_lookup[node.ID] = node
    

    @staticmethod
    def _print_module_times(module_names: list[str], module_times: list[float], pass_name: str):
        """Prints the time taken (absolute, percentage) by each module in forward/backward pass.
        
        Args:
            module_names: List of module names that were run during the forward/backward pass.
            module_times: List of time taken by each module during the forward/backward pass.
            pass_name: Name of the pass. Either 'Forward' or 'Backward'.
        """
        total_time_taken = sum(module_times)
        max_name = max(len(name) for name in module_names)

        print(f'\n{pass_name} Time\n' + '-' * 50)
        for module, t in zip(module_names, module_times):
            print(f'{red}{module:{max_name}}{color_end} ran in: {t:.6f} s' \
                  f' {cyan}[{(t / total_time_taken) * 100:05.2f}%]{color_end}.')
        
        print(f'{red}Total Time{color_end}: {total_time_taken:.4f} s.')
        print()
    

    def forward(self, verbose = False) -> None:
        """Runs the forward method of each module in topological order to perform the neural net
        forward pass.
        
        Args:
            verbose: Flag that decides whether to print the time taken by each module to run.
        """
        if not verbose:
            for node in self.topological_order:
                if isinstance(node, Module):
                    node.forward()
        else:
            module_names = []
            module_times = []

            for node in self.topological_order:
                if isinstance(node, Module):
                    module_names.append(node.ID)
                    start_time = time.perf_counter()
                    # Run the module's forward method
                    node.forward()
                    module_times.append(time.perf_counter() - start_time)
            
            self._print_module_times(module_names, module_times, pass_name = 'Forward')
    

    def backward(self, verbose = False) -> None:
        """Runs the backward method of each module in reverse topological order to perform the
        neural net backward pass.
        
        Args:
            verbose: Flag that decides whether to print the time taken by each module to run.
        """
        if not verbose:
            for node in reversed(self.topological_order):
                if isinstance(node, Module):
                    node.backward()
        else:
            module_names = []
            module_times = []

            for node in reversed(self.topological_order):
                if isinstance(node, Module):
                    module_names.append(node.ID)
                    start_time = time.perf_counter()
                    # Run the module's backward method
                    node.backward()
                    module_times.append(time.perf_counter() - start_time)
            
            self._print_module_times(module_names, module_times, pass_name = 'Backward')
    

    def forward_from_node(self, node: Data) -> None:
        """Runs the forward method starting from a particular node of each module afterwards in
        topological order.
        
        Args:
            node: Data object from which to start the forward pass.
        
        Raises:
            ValueError: If forward pass is initialized at a module instead of a Data object.
        """
        if not isinstance(node, Data):
            raise ValueError('Forward pass can only be initialized starting from a Data object.')
        
        node_idx = self.topological_order.index(node)

        for node in self.topological_order[node_idx: ]:
            if isinstance(node, Module):
                node.forward()


    def update(self) -> None:
        """Updates all the parameters (non-frozen) of each node (Data/Module) in the neural
        network."""
        for node in self.topological_order:
            node.update()


    def freeze(self) -> None:
        """Freezes all the modules (only, not non-frozen Data objects) of the neural network."""
        self.is_frozen = True
        self.optimizer_details = None

        for node in self.topological_order:
            if isinstance(node, Module):
                node.freeze()


    def unfreeze(self, optimizer_details: dict = DEFAULT_OPTIMIZER_DETAILS) -> None:
        """Unfreezes all the modules (only, not frozen Data objects) of the neural network."""
        self.is_frozen = False
        self.optimizer_details = optimizer_details

        for node in self.topological_order:
            if isinstance(node, Module):
                node.unfreeze(optimizer_details = optimizer_details)


    def clear_grads(self) -> None:
        """Clears gradients for each node (Data/Module) in the neural network."""
        for node in self.topological_order:
            node.clear_grads()
    

    def set_learning_rate(self, learning_rate: float) -> None:
        """Sets learning rate for each node (Data/Module) of the neural network."""
        self.learning_rate = learning_rate
        for node in self.topological_order:
            node.set_learning_rate(learning_rate)
    

    def set_optimizer(self, optimizer_details: dict) -> None:
        """Sets optimizer for all the modules (only, not non-frozen Data objects) of the neural
        network.
        
        Raises:
            ValueError: If the net is frozen.
        """
        if self.is_frozen:
            raise ValueError("Net is frozen. Cannot set optimizer in frozen state. Unfreeze first.")
        
        self.optimizer_details = optimizer_details
        for node in self.topological_order:
            if isinstance(node, Module):
                node.set_optimizer(optimizer_details)


    def set_regularization(self, regularizer_details) -> None:
        """Sets regularizer details for all the regularized modules (only, not regularized Data
        objects) of the neural network."""
        self.is_regularized = True
        self.regularizer_details = regularizer_details

        for node in self.topological_order:
            if isinstance(node, Module):
                if node.is_regularizable:
                    node.set_regularization(regularizer_details)


    def set_mode(self, mode: str) -> None:
        """Sets the mode (training/testing) for the neural network.
        
        Args:
            mode: String determining whether the network is in 'train' or 'test' mode.
        
        Raises:
            ValueError: If the specified mode is not available.
        """
        if mode not in self.available_modes:
            raise ValueError(f'Specified mode is not available. Choose from {self.available_modes}')

        self.mode = mode
        for node in self.topological_order:
            if isinstance(node, Module):
                if node.different_at_train_test:
                    node.set_mode(mode)
    

    def visualize(self, filename = 'Source', orientation = 'LR') -> None:
        """Draws a graph to visualize the network."""

        available_orientations = ['LR', 'RL', 'BT', 'TB']
        if orientation not in available_orientations:
            raise ValueError(f'Specified orientation not available. Choose from - {available_orientations}')

        dag = self.graph

        dot_string = 'digraph G {\n'
        dot_string += f'    rankdir={orientation};\n'  # Set direction to left to right

        # Add nodes with their IDs and shapes
        for node, connected_nodes in dag.items():
            # Use ellipse to represent Data nodes and rectangles for Modules
            shape = 'ellipse' if isinstance(node, Data) else 'rect'
            dot_string += f'    {node.ID} [label="{node.ID}", shape = "{shape}"];\n'
            for connected_node in connected_nodes:
                shape = 'ellipse' if isinstance(connected_node, Data) else 'rect'
                dot_string += f'    {connected_node.ID} [label="{connected_node.ID}", shape = "{shape}"];\n'
        
        # Add edges
        for node, connected_nodes in dag.items():
            for connected_node in connected_nodes:
                dot_string += f'    {node.ID} -> {connected_node.ID};\n'
        
        dot_string += '}'

        # Create a Source object and display the graph
        filename += '.gv'
        source = Source(dot_string, filename = filename)
        source.view()
    

    def automate_training(
        self,
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

        # Set net object to self
        net = self

        # Get input, target label and loss node
        x = net.topological_order[0]
        for node in reversed(self.topological_order):
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
    

    def save(self, file_path: str) -> None:
        file_path = Net._ensure_extension(file_path)
        
        # Save the model using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    

    @staticmethod
    def load(file_path: str) -> None:
        file_path = Net._ensure_extension(file_path)
        
        # Save the model using pickle
        with open(file_path, 'rb') as f:
            net = pickle.load(f)
        
        return net
    
    
    @staticmethod
    def _ensure_extension(file_path: str) -> None:
        # Check if the .df extension is already present, if not, then add it
        if not file_path.endswith('.df'):
            file_path += '.df'
        
        return file_path


    def connect(self, root_nodes: list[Data]) -> None:
        self.root_nodes = root_nodes

        # Clear graphs and topological order
        self.graph = {}
        self.graph_visual = {}
        self.topological_order = []
        self.node_lookup = {}

        # Recreate graph and topological order
        self.topological_sort()
        self.create_graph_dicts()
        self.create_lookup()
    

    def disconnect(self, data_obj: Data, module_obj: Module) -> None:
        """Disconnects a data object and module.
        
        Disconnection makes sense only for a data object which is an input to a module. The graph
        resulting from disconnecting a module and an output data object has no use case hence that
        is not taken care of.
        """

        if data_obj in module_obj.inputs:
            data_obj.outputs = []
            self.root_nodes = [data_obj]
        else:
            raise ValueError('Data object provided is not in the list of module inputs.')
        
        # Clear graphs and topological order
        self.graph = {}
        self.graph_visual = {}
        self.topological_order = []
        self.node_lookup = {}

        # Recreate graph and topological order
        self.topological_sort()
        self.create_graph_dicts()
        self.create_lookup()
    

    def share_parameters(self, modules: list[Module]):
        
        # First verify that all modules are of the same type
        module1 = modules[0]
        for module in modules[1:]:
            if not isinstance(module, type(module1)):
                raise TypeError(f'Module {module.ID} is not of same type as {module1.ID} (type: {type(module1).__name__}).')
        
        # Share parameters across the modules
        for module in modules[1:]:
            for param_mod1, param_mod in zip(module1.parameter_list, module.parameter_list):
                param_mod.val = param_mod1.val
                param_mod.deriv = param_mod1.deriv