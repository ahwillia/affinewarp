"""
Tridiagonal matrix algorithm
"""

import numpy as np
from numba import jit, f8

__all__ = ['trisolve']


def trisolve(a, b, c, d):
    '''Solves A x = d for x where A is a tridiagonal matrix.

    Parameters:
        a (ndarray) : vector, lower-diagonal entries of A
        b (ndarray) : vector, diagonal entries of A
        c (ndarray) : vector, upper-diagonal entries of A
        d (ndarray) : matrix, right hand side.

    Returns:
        x (ndarray) : solution, overwrites d

    Reference:
        https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    '''

    # check inputs
    if a.ndim != 1 or b.ndim != 1 or c.ndim != 1:
        raise ValueError('Inputs a, b, and c must be vectors.')
    elif len(a) != len(c):
        raise ValueError('Inputs a and c must have the same length.')
    elif len(b) != (len(a) + 1):
        raise ValueError('Input b must have one more element than a and c.')
    elif len(d) != len(b):
        raise ValueError('Input d must have same length as b.')

    # input d must be a 2d matrix
    if d.ndim == 1:
        d = d[:, None]
    elif d.ndim > 2:
        raise ValueError('Input d has too many dimensions')

    # preallocate space for result
    cp = np.empty(d.shape[0])
    dp = np.empty(d.shape)

    # optimized solver
    return _trisolve(a, b, c, d, cp, dp)


@jit(f8[:, :](f8[:], f8[:], f8[:], f8[:, :], f8[:], f8[:, :]), nopython=True)
def _trisolve(a, b, c, d, cp, dp):

    # number of equations
    n = len(b)

    # initialize forward pass
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    # do forward pass
    for i in range(1, n-1):
        denom = b[i] - a[i-1]*cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1]*dp[i-1]) / denom

    # last element in forward pass
    dp[-1] = (d[-1] - a[-1]*dp[-2]) / (b[-1] - a[-1]*cp[-2])

    # do backward pass
    for i in range(n-2, -1, -1):
        dp[i] = dp[i] - cp[i]*dp[i+1]

    return dp
