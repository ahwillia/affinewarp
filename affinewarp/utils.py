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
