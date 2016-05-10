
import numpy as np

# Author: Jean Kossaifi <jean.kossaifi@gmail.com>

def reflective_correlation_coefficient(y_true, y_pred):
    """Reflective variant of Pearson's product moment correlation coefficient
    where the predictions are not centered around their mean values.

    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Returns
    -------
    float: reflective correlation coefficient
    """
    return np.sum(y_true*y_pred)/np.sqrt(np.sum(y_true**2)*np.sum(y_pred**2))


def mean_squared_error(y_true, y_pred):
    """Returns the mean squared error between the two predictions
    
    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Returns
    -------
    float
    """
    return np.mean((y_true - y_pred)**2)


def RMSE(y_true, y_pred):
    """Returns the regularised mean squared error between the two predictions
    (the square-root is applied to the mean_squared_error)
    
    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Returns
    -------
    float
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def covariance(y_true, y_pred):
    return np.mean((y_true - np.mean(y_true))*(y_pred - np.mean(y_pred)))

def variance(y):
    return covariance(y, y)

def standard_deviation(y):
    return np.sqrt(variance(y))

def correlation(y_true, y_pred):
    return covariance(y_true, y_pred)/np.sqrt(variance(y_true)*variance(y_pred))

def intra_class_correlation(y_true, y_pred):
    """Intra class correlation coefficient ICC(3, 1) for two two raters


    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Reference
    ---------
    Intraclass correlations: Uses in assessing rater reliability.
    Shrout, Patrick E.; Fleiss, Joseph L.
    Psychological Bulletin, Vol 86(2), Mar 1979, 420-428.
    http://dx.doi.org/10.1037/0033-2909.86.2.420

    Returns
    -------
    float
    """
    normalising_factor = variance(y_true) + variance(y_pred)
    
    numerator = 2*covariance(y_true, y_pred)
    if normalising_factor != 0:
        return numerator/normalising_factor
    else:
        return numerator

