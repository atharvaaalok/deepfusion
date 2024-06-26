from .sgd import SGD
from .momentum import Momentum
from .rmsprop import RMSprop
from .adam import Adam
from .adagrad import AdaGrad


def get_optimizer(data_shape, optimizer_details):

    try:
        optimizer = _optimizers_dict[optimizer_details['optimizer_name']]
    except KeyError:
        raise ValueError(f'Specified optimizer not available. Choose from - {list(_optimizers_dict)}')
    else:
        hyperparameters = optimizer_details['hyperparameters']
        return optimizer(data_shape, **hyperparameters)


_optimizers_dict = {
    'SGD': SGD,
    'Momentum': Momentum,
    'AdaGrad': AdaGrad,
    'RMSprop': RMSprop,
    'Adam': Adam,
}


# Set Adam as the default optimizer with default values for the hyperparameters
DEFAULT_OPTIMIZER_DETAILS = {'optimizer_name': 'Adam', 'hyperparameters': {}}