from .sgd import SGD


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
}