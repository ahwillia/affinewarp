import numpy as np
from numba import jit
import sparse


# def modf(U):
#     """Return the fractional and integral parts of an array, element-wise.

#     Note:
#         nearly the same as numpy.modf but returns integer type for i

#     Parameters
#     ----------
#     U : ndarray of floats

#     Returns
#     -------
#     i : ndarray of int32 (integral parts)
#     lam : ndarray of floats (fractional parts)
#     """

#     i = U.astype(np.int32)
#     lam = U % 1

#     return lam, i


def _diff_gramian(T, lam):
    DtD = np.ones((3, T))

    DtD[-1] = 6.0
    DtD[-1, 1] = 5.0
    DtD[-1, 0] = 1.0
    DtD[-1, -1] = 1.0
    DtD[-1, -2] = 5.0

    DtD[-2] = -4.0
    DtD[-2, 1] = -2.0
    DtD[-2, -1] = -2.0

    return DtD * lam
