import numpy as np
from numba import jit, void, f8, int32
import scipy.sparse


def bin_count_data(data, nbins):
    """Bin sparse count time series
    """

    # view input data as numpy array
    data = np.asarray(data)

    # check sparse array inputs
    #   - first check that all matrices are sparse
    #   - second check that all matrices have same shape
    if not (
        data.ndim == 1 and
        all([scipy.sparse.issparse(m) for m in data]) and
        len(set([m.shape for m in data])) == 1
    ):
        raise ValueError('Input must be a list of sparse matrices '
                         'with matching dimensions.')

    # data dimensions
    n_trials = data.shape[0]
    n_features = data[0].shape[1]
    binned = np.empty((n_trials, nbins, n_features), dtype=data[0].dtype)

    # edges for the time bins
    edges = np.linspace(0, data[0].shape[0]-1, nbins+1)
    edges[0], edges[-1] = -1, data[0].shape[0]

    for k in range(n_trials):
        T, N = data[k].nonzero()
        for n in range(n_features):
            binned[k, :, n] = np.diff(np.searchsorted(T[N == n], edges))

    return binned


def spiketimes_per_neuron(data):
    """Convert from list of sparse matrices to list of spike times.
    """

    trials, times, neurons = [], [], []
    for k, x in enumerate(data):
        t, n = x.nonzero()
        idx = np.argsort(t)
        trials += np.full(len(t), k).tolist()
        neurons += n[idx].tolist()
        times += (t[idx] / x.shape[0]).tolist()

    trials = np.array(trials)
    times = np.array(times)
    neurons = np.array(neurons)

    num_neurons = data[0].shape[1]
    K = [trials[neurons == n] for n in range(num_neurons)]
    T = [times[neurons == n] for n in range(num_neurons)]

    return K, T


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





# _reduce_sum_assign(_WtW[1, :], i, one_m_lam**2)
# _reduce_sum_assign(_WtW[1, :], i+1, lam**2)
# _reduce_sum_assign(_WtW[0, 1:], i, lam*one_m_lam)

# _reduce_sum_assign(WtX, i, (1-lam[:, None]) * data.reshape(-1, N))
# _reduce_sum_assign(WtX, i+1, lam[:, None] * data.reshape(-1, N))

# def _warping_normal_eqs(WtW, WtX, data, X, Y):

#     # iterate over trials
#     for i in range(len(X)):

#         # initialize line segement for interpolation
#         y0 = Y[i, 0]
#         x0 = X[i, 0]
#         slope = (Y[i, 1] - Y[i, 0]) / (X[i, 1] - X[i, 0])

#         # 'm' counts the timebins within trial 'i'.
#         # 'n' counts knots in piecewise affine warping function.
#         m = 0
#         n = 1

#         # iterate over all time bins.
#         while (m < T):

#             # update interpolation point
#             while (n < N-1) and (m/(T-1) > X[i, n]):
#                 y0 = Y[i, n]
#                 x0 = X[i, n]
#                 slope = (Y[i, n+1] - y0) / (X[i, n+1] - x0)
#                 n += 1

#         # initialize line segement for interpolation
#         y0 = Y[i, 0]
#         x0 = X[i, 0]
#         slope = (Y[i, 1] - Y[i, 0]) / (X[i, 1] - X[i, 0])

#     for k in trials:
#         wfunc, Xk = self.warping_funcs[k], data[k]
#         lam, i = modf(wfunc * (T-1))

#         _reduce_sum_assign(_WtW[1, :], i, (1-lam)**2)
#         _reduce_sum_assign(_WtW[1, :], i+1, lam**2)
#         _reduce_sum_assign(_WtW[0, 1:], i, lam*(1-lam))

#         _reduce_sum_assign_matrix(WtX, i, (1-lam[:, None]) * Xk)
#         _reduce_sum_assign_matrix(WtX, i+1, lam[:, None] * Xk)
