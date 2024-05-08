from ..data.data import Data
from ..modules.module import Module
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS


class Net:

    available_modes = ['train', 'test']

    def __init__(
        self,
        ID,
        root_nodes,
        optimizer_details = DEFAULT_OPTIMIZER_DETAILS,
        learning_rate = 1e-6,
        is_regularized = False,
        regularizer_details = None,
    ) -> None:
        
        self.ID = ID
        
        self.root_nodes = root_nodes

        self.graph = {}
        self.graph_visual = {}
        self.topological_order = []
        self.node_lookup = {}

        self.is_frozen = False
        self.optimizer_details = optimizer_details if not self.is_frozen else None

        self.learning_rate = learning_rate

        self.is_regularized = is_regularized
        self.regularizer_details = regularizer_details if is_regularized else None

        self.mode = 'train'
        
        self.run_setup()
    

    def run_setup(self):
        self.topological_sort()
        self.create_graph_dicts()
        self.create_lookup()

        if self.is_regularized:
            self.set_regularization(self.regularizer_details)
        self.set_optimizer(self.optimizer_details)


    def topological_sort(self):
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
            
        _topological_sort_util(*self.root_nodes)
        self.topological_order = topological_order
    

    def create_graph_dicts(self):
        for node in self.topological_order:
            if isinstance(node, Data):
                self.graph[node] = node.outputs
                self.graph_visual[node.ID] = [child.ID for child in node.outputs]
            elif isinstance(node, Module):
                self.graph[node] = [node.output]
                self.graph_visual[node.ID] = [node.output.ID]


    def create_lookup(self):
        for node in self.topological_order:
            self.node_lookup[node.ID] = node
    

    def forward(self):
        for node in self.topological_order:
            if isinstance(node, Module):
                node.forward()


    def backward(self):
        for node in reversed(self.topological_order):
            if isinstance(node, Module):
                node.backward()


    def update(self):
        for node in self.topological_order:
            node.update()


    def freeze(self):
        self.is_frozen = True
        self.optimizer_details = None

        for node in self.topological_order:
            if isinstance(node, Module):
                node.freeze()


    def unfreeze(self, optimizer_details = DEFAULT_OPTIMIZER_DETAILS):
        self.is_frozen = False
        self.optimizer_details = optimizer_details

        for node in self.topological_order:
            if isinstance(node, Module):
                node.unfreeze(optimizer_details = optimizer_details)


    def clear_grads(self):
        for node in self.topological_order:
            node.clear_grads()
    

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for node in self.topological_order:
            node.set_learning_rate(learning_rate)
    

    def set_optimizer(self, optimizer_details):
        if not self.is_frozen:
            for node in self.topological_order:
                if isinstance(node, Module):
                    node.set_optimizer(optimizer_details)


    def set_regularization(self, regularizer_details):
        self.is_regularized = True
        self.regularizer_details = regularizer_details

        for node in self.topological_order:
            if isinstance(node, Module):
                if node.is_regularizable:
                    node.set_regularization(regularizer_details)


    def set_mode(self, mode):

        if mode not in self.available_modes:
            raise ValueError(f'Specified mode is not available. Choose from {self.available_modes}')

        self.mode = mode
        for node in self.topological_order:
            if isinstance(node, Module):
                if node.different_at_train_test:
                    node.set_mode(mode)