"""Banded Matrix Utilities and Helper Functions."""
from scipy.linalg import eig_banded, solveh_banded
import numpy as np
import numba


def nnls_solveh_banded(S, B, X0, niter=50):

    # Number of timebins and units.
    nt = S.shape[1]
    nn = B.shape[1]

    # On first iteration, warm-start from projected least-squares.
    if X0 is None:
        X0 = np.maximum(0, solveh_banded(S, B))

    # Determine Lipshitz constant
    w = eig_banded(
        S, lower=False, select='i',
        select_range=(nt - 1, nt - 1),
        eigvals_only=True)[0]

    # Run projected gradient descent in parallel across neurons.
    _parallel_proj_grad(S, B, X0, 1 / w, niter)

    return X0


@numba.jit(nopython=True, parallel=True)
def _parallel_proj_grad(S, B, X0, stepsize, niter):
    for i in numba.prange(X0.shape[-1]):
        nnls_proj_grad(S, B[:, i], X0[:, i], stepsize, niter)


@numba.jit(nopython=True)
def nnls_proj_grad(S, b, x, stepsize, niter):

    nn = x.size
    Sx = np.empty_like(x)

    for itr in range(niter):

        # Update banded matrix multiply
        sym_bmat_mul(S, x, Sx)

        # Projected gradient descent step.
        for i in range(nn):
            x[i] = x[i] + stepsize * (b[i] - Sx[i])
            if x[i] < 0:
                x[i] = 0.0


@numba.jit(nopython=True)
def sym_bmat_mul(S, x, out):
    """
    Symmetric banded matrix times vector.

    Parameters
    ----------
    S : ndarray
        (b x n) array specifying symmetric banded matrix in
        "upper form".
    x : ndarray
        Vector of length n, multiplying S.
    out : ndarray
        Vector of length n, which is overwritten to store result.
    """

    b, n = S.shape

    for i in range(n):
        out[i] = S[-1, i] * x[i]

    for j in range(1, b):
        for i in range(n - j):
            out[i] += S[(-j - 1), j + i] * x[j + i]
            out[j + i] += S[(-j - 1), j + i] * x[i]
