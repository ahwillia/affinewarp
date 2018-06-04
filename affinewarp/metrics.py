import numpy as np
from numbers import Integral
from .spikedata import bin_spikes


def mean_noise_var(data, bins):

    if isinstance(bins, Integral):
        bins = np.asarray([bins])
    else:
        bins = np.asarray(bins)

    if not np.issubdtype(bins.dtype, np.integer):
        raise ValueError('bins must be specified as integer')

    result = []
    for b in bins:
        binned = bin_spikes(data, b)
        result.append(binned.var(axis=0).mean(axis=0))
    return np.array(result)


def calc_snr(data, bins):

    if isinstance(bins, Integral):
        bins = np.asarray([bins])
    else:
        bins = np.asarray(bins)

    if not np.issubdtype(bins.dtype, np.integer):
        raise ValueError('bins must be specified as integer')

    result = []
    for b in bins:
        binned = bin_spikes(data, b)
        m = binned.mean(axis=0)
        s = binned.std(axis=0)
        result.append(np.ptp(m, axis=0) / np.max(s, axis=0))
    return np.array(result)


def participation(M):
    """Participation ratio
    """
    lam = np.linalg.svd(M, compute_uv=False)**2
    return np.sum(lam)**2 / np.sum(lam**2)
