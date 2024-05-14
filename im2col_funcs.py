import numpy as np


def im2col(X, filter_size, padding, stride):
    # Pad the input
    X = np.pad(X, pad_width = ((0, 0), (padding, padding), (padding, padding)))

    # Determine the output size
    D, H, W = X.shape
    o = (H - filter_size) // stride + 1

    X_flat = np.zeros((filter_size * filter_size * D, o * o))

    col = 0
    for i in range(o):
        for j in range(o):
            idx_i, idx_j = i * stride, j * stride
            X_patch = X[:, idx_i: idx_i + filter_size, idx_j: idx_j + filter_size]
            X_flat[:, col: col + 1] = X_patch.reshape(-1, 1)
            col += 1
    
    return X_flat


def col2im(X_deriv_flat, X_shape, filter_size, padding, stride):

    D, H, W = X_shape

    # Set the derivative size
    X_deriv = np.zeros((D, H + 2 * padding, W + 2 * padding))

    o = (H + 2 * padding - filter_size) // stride + 1

    col = 0
    for i in range(o):
        for j in range(o):
            idx_i, idx_j = i * stride, j * stride
            X_deriv_flat_col = X_deriv_flat[:, col: col + 1]
            X_deriv_flat_col_shaped = X_deriv_flat_col.reshape(D, filter_size, filter_size)
            X_deriv[:, idx_i: idx_i + filter_size, idx_j: idx_j + filter_size] += X_deriv_flat_col_shaped
            col += 1
    
    # Remove padding
    X_deriv = X_deriv[:, padding: padding + H, padding: padding + W]
    
    return X_deriv