from .zigzag import zigzag_scan
from scipy import fftpack

def dct2(image):
    """ Get the 2D Cosine Transform of a gray scale Image

    Parameters
    ----------
    image: ndarray of shape (width, height)

    Returns
    -------
    2D-DCT of the input image
    """
    return fftpack.dct(fftpack.dct(image, norm='ortho').T, norm='ortho').T

def dct2_features(image, n_features):
    """Returns the first `n_coefs` coefficients of the 2-dimensional DCT following the zigzag scheme.

    Paramters
    ---------
    image: ndarray of shape (width, height)
    n_features: int
        size of the descriptor

    Returns
    -------
    features: ndarray of shape (n_features, )
    """
    dct_image = dct2(image)
    features = zigzag_scan(dct_image, n_features)
    return features
