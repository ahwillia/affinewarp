import numpy as np
from numbers import Integral
from .spikedata import bin_spikes
from .utils import check_data_tensor


def snr(data, nbins=None):
    """
    Signal-to-Noise Ratio (SNR) for each neuron.
    """
    binned = _bin_data(data, nbins)
    m = binned.mean(axis=0)
    signal = np.abs(m - m.mean(0)).max(0)
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


def _bin_data(data, nbins):

    # check input
    data, is_spikes = check_data_tensor(data)

    # if spiking data is provided, bin spike times
    if is_spikes and nbins is None:
        raise ValueError(
            'If data is provided in a spike data format, number of bins '
            '(`nbins`) must be specified.'
        )

    # bin spikes and return
    elif is_spikes:
        return bin_spikes(data, nbins)

    # data is in dense array format, return it as is.
    else:
        return data
