import numpy as np


def calc_snr(binned):
    """Signal-to-noise ratio
    """
    m = binned.mean(axis=0)
    s = binned.std(axis=0)

    return np.ptp(m, axis=0) / np.max(s, axis=0)
