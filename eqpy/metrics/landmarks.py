# Author: Jean Kossaifi

""" A module implementing various metrics for measuring the accuracy of
landmark detection
"""

import numpy as np
from scipy.linalg import norm


def max_face_size(shape):
    """
    Returns the size of the face

    Parameters
    ----------
    shape: ndarray of shape (n_points, 2)
    """
    return np.sum(shape.max(axis=0) - shape.min(axis=0))/2


def interocular_distance(shape):
    """ Returns the interocular distance

    Parameters
    ----------
    shape: ndarray of shape (n_points, 2)
    """
    n_points, n_dims = shape.shape
    if n_points == 68:
        left_corner = 36
        right_corner = 45
    elif n_points == 51 or n_points == 49:
        left_corner = 19
        right_corner = 28

    return norm(shape[left_corner, :] - shape[right_corner, :], 2)


def pt_pt_error(ground_truth_shape, predicted_shape, normalize='max_face_size', normalization_shape=None):
    """ Point to point error between the two given shapes

    Parameters
    ----------
    ground_truth_shape: np-array of shape [n_points, 2]
            ground truth shape
    predicted_shape: np-array of shape [n_points, 2]
        predicted shape
    normalize: string or float, default is 'max_face_size'
        {'max_face_size', 'interocular'}
        normalising factor for the error
    normalization_shape: None or np array of shape [n_points, 2], default is None
        if not None, the normalization factor will be computed on that shape
        
    Returns
    -------
    float: point to point error
    """
    ground_truth_shape = ground_truth_shape.reshape((-1, 2))
    predicted_shape = predicted_shape.reshape((-1, 2))
    
    if normalization_shape is None:
        normalization_shape = ground_truth_shape
    
    if normalize is None:
        normalizing_factor = 1
    elif normalize == 'max_face_size':
        normalizing_factor = max_face_size(normalization_shape)
    elif normalize == 'interocular':
        normalizing_factor = interocular_distance(normalization_shape)
    elif type(normalize) is float:
        normalizing_factor = normalize

    per_point_error = np.sqrt(np.sum((ground_truth_shape - predicted_shape)**2, axis=-1))
    return np.mean(per_point_error) / normalizing_factor


def cumulative_errors(errors, max_error=None, n_bins=50):
    """Computes the cumulative function of the errors repartition

    Parameters
    ----------
    errors: np.array of shape (n_errors, )
        the errors
    max_errors: int, default is None
        if None the maximum is computed from the errors
        otherwise the cumulative errors will be limited to max_error
    n_bins: int, default is 50
        number of bins for the computation of the cumulative error
    bin_precision: None or int, default is None
        if not None, the bins returned will have a float precision of bin_precision.
        Note that this is done after calculations and do not affect computation of
        the cumulative errors.

    Returns
    -------
    bins: array of shape (n_bins, )
        the max value of accepted error for each bin
    cum_errors: array of shape (n_bins, )
        cum_error[i] corresponds to the percentage of errors lower or equal
        than bins[i] 

    Note
    ----
    0 <= cum_err[i] <= 1
    """
    if not max_error:
        max_error = np.max(errors)

    n_errors = len(errors)
    bins = np.linspace(0, max_error, n_bins)
    cum_errors = np.zeros(n_bins)
    for index in range(n_bins):
        cum_errors[index] = np.sum(errors < bins[index], dtype=np.float)/n_errors

    return bins, cum_errors
