import numpy as np
import sparse
from numba import jit


def _check_spike_tuple(data):
    """Check that spike indices in tuple formate are correct
    """

    if not (
        len(data) == 3 and
        len(np.unique([c.size for c in data])) == 1 and
        all(np.issubdtype(c.dtype, np.integer) for c in data)
    ):
        raise ValueError(
            'Spiking data supplied as a tuple must be 3 arrays holding '
            'integers. Arrays must be equal length specifying trial number, '
            'spike time, neuron id (in that order).')


def is_spikedata(data):
    """Returns True if data can be interpreted as spiking data encoded as
    a sparse 3d tensor. Returns False if data is a dense array.
    """
    if isinstance(data, sparse.COO) and data.ndim == 3:
        return True

    try:
        _check_spike_tuple(data)
        return True

    except ValueError:
        return False


def get_spike_coords(data):
    """Returns indices of spike times (trial number, time bin, neuron id)
    """

    if isinstance(data, sparse.COO):
        if data.ndim != 3:
            raise ValueError('Spiking data supplied as a sparse array '
                             'must have ndim == 3.')
        return data.coords

    elif isinstance(data, tuple):
        _check_spike_tuple(data)
        return data

    else:
        raise ValueError('Spiking data must be supplied as tuple or sparse '
                         'array.')


def get_spike_shape(data):
    """Returns shape of spike data sparse array
    """

    if hasattr(data, 'shape'):
        return data.shape

    elif isinstance(data, tuple):
        _check_spike_tuple(data)
        return tuple([d.max()+1 for d in data])


def bin_spikes(data, nbins, shape=None):
    """Bin spike data into dense 3d tensor (trials x nbins x neurons)
    """

    # get spike indices
    trials, times, neurons = get_spike_coords(data)

    try:
        shape = data.shape
    except AttributeError:
        shape = [trials.max()+1, times.max()+1, neurons.max()+1]

    n_trials = shape[0]
    n_timepoints = shape[1]
    n_neurons = shape[2]

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


def calc_snr(data, nbins=None):
    """Compute signal-to-noise estimate for each neuron.

    Returns
    -------
    1d array, SNR for each neuron
    """

    # bin spike before computing SNR
    if nbins is None:
        binned = data
    else:
        binned = bin_spikes(data, nbins)

    m = binned.mean(axis=0)
    s = binned.std(axis=0)
    return m.ptp(axis=0) / s.max(axis=0)


@jit(nopython=True)
def _bin_assign(out, trials, bin_ind, neurons):
    """Add spikes into bins
    """
    for i in range(len(trials)):
        out[trials[i], bin_ind[i], neurons[i]] += 1
