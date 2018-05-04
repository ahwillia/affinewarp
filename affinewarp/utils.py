import numpy as np
from numba import jit


def participation(M):
    lam = np.linalg.svd(M, compute_uv=False)**2
    return np.sum(lam)**2 / np.sum(lam**2)


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


@jit(nopython=True)
def _reduce_sum_assign(U, i, elems):
    n = len(U)
    for j in range(len(i)):
        if i[j] < n:
            U[i[j]] += elems[j]


@jit(nopython=True)
def _fast_template_grams(WtW, WtX, unf, lam, i):
    n = len(WtX)
    for j in range(len(i)):
        mlam = 1-lam[j]
        WtW[1, i[j]] += mlam**2
        WtX[i[j]] += mlam * unf[j]
        if i[j] < (n-1):
            WtW[1, i[j]+1] += lam[j]**2
            WtW[0, i[j]+1] += mlam * lam[j]
            WtX[i[j]+1] += lam[j] * unf[j]


def quad_loss(pred, targ):
    """Row-wise mean squared Euclidean error between prediction and target array
    """
    return np.mean(((pred - targ)**2).reshape(pred.shape[0], -1), axis=-1)
