import numpy as np
from affinewarp.bmat import sym_bmat_mul
from numpy.testing import assert_allclose


def test_sym_mul():

    S = np.array([
        [1, 2, 1, 0, 0, 0, 0, 0],
        [2, 1, 1, 4, 0, 0, 0, 0],
        [1, 1, 5, 1, 1, 0, 0, 0],
        [0, 4, 1, 4, 1, 8, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 8, 1, 7, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 3],
        [0, 0, 0, 0, 0, 1, 3, 9],
    ]).astype('float')

    x = np.random.randn(8)

    Sb = np.full((3, 8), np.nan)
    Sb[-1] = np.diag(S)
    Sb[-2, 1:] = np.diag(S, 1)
    Sb[-3, 2:] = np.diag(S, 2)

    y = np.dot(S, x)

    z = np.empty_like(x)
    sym_bmat_mul(Sb, x, z)

    assert_allclose(z, y)
