import numpy as np
from numpy.testing import assert_almost_equal
from ..regression import correlation, intra_class_correlation, standard_deviation

def test_correlation():
    a = np.random.random(10)
    b = np.random.random(10)
    assert_almost_equal(correlation(a, a*2+1), 1)
    assert_almost_equal(correlation(a, -a*2+1), -1)
    
    a = np.array([1, 2, 3, 2, 1])
    b = np.array([1, 2, 3, 4, 5])
    assert_almost_equal(correlation(a, b), 0)

    
def test_intra_class_correlation():
    """Test the test_intra_class_correlation
    
    We know that the ICC of two samples normalised by their std
    is equal to the pearson correlation coeffient
    """
    a = np.random.random(10)
    b = np.random.random(10)
    na = a/standard_deviation(a)
    nb = b/standard_deviation(b)
    corr = correlation(a, b)
    icc_norm = intra_class_correlation(na, nb)
    assert_almost_equal(corr, icc_norm)
        

if __name__ == '__main__':
    test_correlation()
    test_intra_class_correlation()
