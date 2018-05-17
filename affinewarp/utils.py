import numpy as np
from numba import jit
import sparse


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
def _fast_template_grams(WtW, WtX, data, X, Y):

    K, T, N = data.shape
    n_knots = X.shape[1]

    # iterate over trials
    for k in range(len(X)):

        # initialize line segement for interpolation
        y0 = Y[k, 0]
        x0 = X[k, 0]
        slope = (Y[k, 1] - Y[k, 0]) / (X[k, 1] - X[k, 0])

        # 'n' counts knots in piecewise affine warping function.
        n = 1

        # iterate over time bins
        for t in range(T):

            # fraction of trial complete
            x = t / (T - 1)

            # update interpolation point
            while (n < n_knots-1) and (x > X[k, n]):
                y0 = Y[k, n]
                x0 = X[k, n]
                slope = (Y[k, n+1] - y0) / (X[k, n+1] - x0)
                n += 1

            # compute index in warped time
            z = y0 + slope*(x - x0)

            if z >= 1:
                WtX[-1] += data[k, t]
                WtW[1, -1] += 1.0

            elif z <= 0:
                WtX[0] += data[k, t]
                WtW[1, 0] += 1.0

            else:
                i = int(z * (T-1))
                lam = (z * (T-1)) % 1

                WtX[i] += (1-lam) * data[k, t]
                WtW[1, i] += (1-lam)**2
                WtW[1, i+1] += lam**2
                WtW[0, i+1] += (1-lam) * lam
                WtX[i+1] += lam * data[k, t]


@jit(nopython=True)
def _force_monotonic_knots(X, Y):
    K, P = X.shape
    for k in range(K):

        for p in range(P-1):
            x0 = X[k, p]
            y0 = Y[k, p]
            x1 = X[k, p+1]
            y1 = Y[k, p+1]
            dx = X[k, p+1] - X[k, p]
            dy = Y[k, p+1] - Y[k, p]

            if (dx < 0) and (dy < 0):
                # swap both x and y coordinates
                tmp = X[k, p]
                X[k, p] = X[k, p+1]
                X[k, p+1] = tmp
                # swap y
                tmp = Y[k, p]
                Y[k, p] = Y[k, p+1]
                Y[k, p+1] = tmp

            elif dx < 0:
                # swap x coordinate
                tmp = X[k, p]
                X[k, p] = X[k, p+1]
                X[k, p+1] = tmp
                # set y coordinates to mean
                Y[k, p] = Y[k, p] + (dy/2) - 1e-3
                Y[k, p+1] = Y[k, p+1] - (dy/2) + 1e-3

            elif dy < 0:
                # set y coordinates to mean
                Y[k, p] = Y[k, p] + (dy/2) - 1e-3
                Y[k, p+1] = Y[k, p+1] - (dy/2) + 1e-3

        # TODO - redistribute redundant edge knots
        for p in range(P):
            if X[k, p] <= 0:
                X[k, p] = 0.0 + 1e-3*p
            elif X[k, p] >= 1:
                X[k, p] = 1.0 - 1e-3*(P-p-1)

    return X, Y
