import numpy as np
import sparse
from numba import jit


def bin_spikes(data, nbins, tmin=None, tmax=None):

    if isinstance(data, sparse.COO):
        trials, times, neurons = data.coords
    else:
        trials, times, neurons = np.nonzero(data)

    # compute bin size
    n_trials = data.shape[0]
    n_timepoints = data.shape[1]
    n_neurons = data.shape[2]

    # compute bin id for each spike time
    #   - note: dividing by n_timepoints first is actually critical if data
    #           is provided as sparse.COO, which provides coords as uint16,
    #           leading to overflow if multiplication is done first.
    bin_ind = (nbins * (times / n_timepoints)).astype(int)

    # bin spikes
    binned = np.zeros((n_trials, nbins, n_neurons), dtype=int)
    _bin_assign(binned, trials, bin_ind, neurons)

    # return (trials x timebins x neurons) array of binned spike counts
    return binned


@jit(nopython=True)
def _bin_assign(out, trials, bin_ind, neurons):
    """Add spikes into bins
    """
    for i in range(len(trials)):
        out[trials[i], bin_ind[i], neurons[i]] += 1
