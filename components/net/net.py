from ..data.data import Data
from ..modules.module import Module
from ..optimizers import DEFAULT_OPTIMIZER_DETAILS


class Net:

    available_modes = ['train', 'test']

    def __init__(self, ID, optimizer_details = DEFAULT_OPTIMIZER_DETAILS, learning_rate = 1e-6, is_regularized = False, regularizer_details = None):
        self.ID = ID
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
    

    def add_nodes(self, module, inputs, output):
        for input in inputs:
            if input not in self.graph:
                self.graph[input] = [module]
                self.graph_visual[input.ID] = [module.ID]
            else:
                self.graph[input].append(module)
                self.graph_visual[input.ID].append(module.ID)
        
        self.graph[module] = [output]
        self.graph_visual[module.ID] = [output.ID]

        self.graph[output] = []
        self.graph_visual[output.ID] = []
    

    def _topological_sort_util(self, node, visited, stack):
        visited.add(node)
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                self._topological_sort_util(neighbor, visited, stack)
        stack.append(node)
    

    def topological_sort(self):
        visited = set()
        stack = []
        for node in self.graph:
            if node not in visited:
                self._topological_sort_util(node, visited, stack)
        
        self.topological_order = stack[: : -1]
    
    
    def create_lookup(self):
        for node in self.topological_order:
            self.node_lookup[node.ID] = node
    
    
    def run_setup(self):
        self.topological_sort()
        self.create_lookup()
        if self.is_regularized:
            self.set_regularization(self.regularizer_details)
    

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