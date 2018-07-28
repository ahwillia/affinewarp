"""Common metrics for evaluating across-trial variability."""

import numpy as np
from numbers import Integral


def snr(data, nbins=None):
    """
    Signal-to-Noise Ratio (SNR) for each neuron.

    For each neuron, the SNR across trials is defined as the largest deviation
    from the mean firing rate ("the signal") divided by the maximum standard
    deviation of any timebin across trials ("the noise"). Note that this metric
    may be sensitive to the size of the time bins.

    Parameters
    ----------
    data: SpikeData or ndarray
        Multi-trial dataset. If provided as a SpikeData instance, the spike
        counts are binned before computing the SNR. If provided as an ndarray,
        then the input is interpreted as binned spike counts.
    nbins: int (optional)
        If 'data' is a SpikeData object, then 'nbins' must be provided
        in order to compute PSTH.

    Returns
    -------
    snr: ndarray of floats
        SNR for each neuron.
    """
    if isinstance(data, SpikeData) and nbins is None:
        raise ValueError("'nbins' must also be specified if data is in "
                         "SpikeData format.")
    elif isinstance(data, SpikeData):
        binned = data.bin_spikes(data, nbins)
    elif isinstance(data, np.ndarray) and data.ndim == 3:
        binned = data
    else:
        raise ValueError("'data' must be a SpikeData instance or a 3d "
                         "numpy array of binned spike counts with shape"
                         "(trials x timebins x neurons).")

    m = binned.mean(axis=0)
    signal = np.abs(m - m.mean(0)).max(axis=0)
    noise = np.max(binned.std(axis=0), axis=0)
    return signal / noise


def rmse(data, nbins=None):
    """
    Root-Mean-Squared-Error of trial-average for each neuron.
    """
    binned = _bin_data(data, nbins)
    resid = binned - binned.mean(axis=0, keepdims=True)
    return np.sqrt(np.mean(resid ** 2, axis=(0, 1)))


def r_squared(data, nbins=None):
    """
    Coefficient of determination of trial-average for each neuron.
    """
    binned = _bin_data(data, nbins)
    # constant firing rate model
    resid = binned - binned.mean(axis=(0, 1), keepdims=True)
    ss_data = np.sum(resid ** 2, axis=(0, 1))
    # psth model
    resid = binned - binned.mean(axis=0, keepdims=True)
    ss_model = np.sum(resid ** 2, axis=(0, 1))
    # return explained variance
    return 1 - ss_model / ss_data
