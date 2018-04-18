import numpy as np
from numba import jit, void, f8, int32


def modf(U):
    """Return the fractional and integral parts of an array, element-wise.

    Note:
        nearly the same as numpy.modf but returns integer type for i

    Parameters
    ----------
    U : ndarray of floats

    Returns
    -------
    i : ndarray of int32 (integral parts)
    lam : ndarray of floats (fractional parts)
    """

    i = U.astype(np.int32)
    lam = U % 1

    return lam, i


@jit(void(f8[:], int32[:], f8[:]), nopython=True)
def _reduce_sum_assign(U, i, elems):
    n = len(U)
    for j in range(len(i)):
        U[i[j] % n] += elems[j]


@jit(void(f8[:, :], int32[:], f8[:, :]), nopython=True)
def _reduce_sum_assign_matrix(U, i, elems):
    n = len(U)
    for j in range(len(i)):
        U[i[j] % n] += elems[j]
