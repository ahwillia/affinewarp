from scipy.spatial.distance import cdist
import numpy as np
from numba import jit, void, f8, int32
import numba


def dtw_path(x, y, dist):
    """Dynamic time warping alignment of two multi-dimensional time series.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    dist : str
    """
    if np.ndim(x) == 1:
        x = x[:, None]
    if np.ndim(y) == 1:
        y = y[:, None]
    r, c = len(x), len(y)

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    D[1:, 1:] = cdist(x, y, dist)

    i_path = np.empty(r*c, dtype=np.int32)
    j_path = np.empty(r*c, dtype=np.int32)

    c, total_cost = _dtw_forward_backward(D, i_path, j_path)

    i_path = i_path[:c]
    j_path = j_path[:c]

    return i_path, j_path, total_cost


@jit(numba.types.Tuple((int32, f8))(f8[:, :], int32[:], int32[:]), nopython=True)
def _dtw_forward_backward(D, i_path, j_path):

    # dimensions of distance matrix
    ni = D.shape[0]
    nj = D.shape[1]

    # forward pass
    for i in range(ni-1):
        for j in range(nj-1):

            a = D[i, j]
            b = D[i, j+1]
            c = D[i+1, j]

            if a < b and a < c:
                # min(a, b, c) == a
                D[i+1, j+1] += a
            elif b < c:
                # min(a, b, c) == b
                D[i+1, j+1] += b
            else:
                # min(a, b, c) == c
                D[i+1, j+1] += c

    # initialize traceback through matrix
    i = ni-2
    j = nj-2

    i_path[0] = i
    j_path[0] = j

    k = 1
    cost = 0.0

    # iterate until we reach left column or top row
    while (i > 0) or (j > 0):

        # three potential moves
        a = D[i, j]
        b = D[i, j+1]
        c = D[i+1, j]

        # choose move with minimum cost
        if a <= b and a <= c:
            # min(a, b, c) == a
            cost += a
            i -= 1
            j -= 1
        elif b < c:
            # min(a, b, c) == b
            cost += b
            i -= 1
        else:
            # min(a, b, c) == c
            cost += c
            j -= 1

        # record path
        i_path[k] = i
        j_path[k] = j

        # count length of the paths
        k += 1

    return k, cost
