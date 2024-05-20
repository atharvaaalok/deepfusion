from deepfusion.utils.colors import red, cyan, color_end


def print_net_performance(epochs: int, epoch: int, J_train: float, J_val: float = None) -> None:
    # Get num of digits in the number 'epochs' for formatted printing
    num_digits = len(str(epochs))

    # Print the current performance
    print_performance = (
        f'{red}Epoch:{color_end} [{epoch:{num_digits}}/{epochs}].  ' +
        f'{cyan}Train Cost:{color_end} {J_train:11.6f}.  '
    )
    if J_val is not None:
        print_performance += f'{cyan}Val Cost:{color_end} {J_val:11.6f}'
    
    print(print_performance)