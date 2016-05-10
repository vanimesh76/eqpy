import numpy as np

def zigzag_scan(matrix, n_values):
    """Reads the matrix in a jpeg zigzag order

    Parameters
    ----------
    matrix: array of shape (heights, width)
    n_values: number of values to read

    Returns
    -------
    values: list of len n_values containing the first values
        read in jpeg zigzag order

    Notes
    -----
    zigzag order:
    [[ 0  1  5  6 14]
     [ 2  4  7 13 15]
     [ 3  8 12 16 21]
     [ 9 11 17 20 22]
     [10 18 19 23 24]]
    """
    height, width = matrix.shape
    
    values = []
    
    # Current position in the matrix
    i, j = 0, 0 # i = line, j = column
    # Are we going up or down diagonally?
    moving_up = True
    
    while len(values) < n_values:
        values.append(matrix[i, j])

        if j == (width - 1):
            if moving_up:
                i += 1
                moving_up = False
            else: # start going down diagonally
                j -= 1
                i += 1
        elif i == (height - 1):
            if moving_up:
                j += 1
                i -= 1
            else: # start going down diagonally
                j += 1
                moving_up = True
        elif i == 0: # We are touching the upper side
            if moving_up:
                j += 1
                moving_up = False
            else: # start going down diagonally
                j -= 1
                i += 1
        elif j == 0: # We are touching the right side
            if moving_up:
                i -= 1
                j += 1
            else:
                i += 1
                moving_up = True
        elif moving_up:
            i -= 1
            j += 1
        else:
            i += 1
            j -= 1

    return np.array(values)


def zigzag_upper_triangle(n_values): 
    """Reads the matrix in a jpeg zigzag order ONLY FOR THE UPPER LEFT TRIANGLE

    Warning
    -------
    Will work ONLY for the upper left triangle!!

    Parameters
    ----------
    matrix: array of shape (heights, width)
    n_values: number of values to read

    Returns
    -------
    values: list of len n_values containing the first values
        read in jpeg zigzag order

    Notes
    -----
    zigzag order:
    [[ 0  1  5  6 14]
     [ 2  4  7 13 15]
     [ 3  8 12 16 21]
     [ 9 11 17 20 22]
     [10 18 19 23 24]]
    """
    x_indices = [0]
    x_max = 1
    while len(x_indices) < n_values:
        indices = list(range(x_max + 1))
        x_indices += indices + [x_max + 1] + indices[::-1]
        x_max += 2

    y_indices = []
    y_max = 0
    while len(y_indices) < n_values:
        indices = list(range(y_max + 1))
        y_indices += indices + [y_max + 1] + indices[::-1]
        y_max += 2
        
    return x_indices[:n_values], y_indices[:n_values]
