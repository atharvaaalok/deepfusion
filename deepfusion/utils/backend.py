class Backend:
    backend = 'cpu'

    @staticmethod
    def set_backend(backend: str):
        if backend not in ['cpu', 'gpu']:
            raise ValueError('Backend must be "CPU" or "GPU".')
        
        Backend.backend = backend
    

    @staticmethod
    def get_backend():
        return Backend.backend
    

    @staticmethod
    def get_array_module():
        if Backend.backend == 'gpu':
            try:
                import cupy as np
            except ImportError:
                raise ImportError("CuPy is not installed. Please install it to use GPU backend.")
        else:
            import numpy as np
        
        return np