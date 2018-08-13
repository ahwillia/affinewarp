import numba
import numpy as np
from .spikedata import SpikeData
from scipy.interpolate import interp1d


def check_data_tensor(data):
    """
    Check if input is either SpikeData or ndarray.

    Parameters
    ----
    arr : ndarray or SpikeData instance
        input data array

    Returns
    -------
    arr : ndarray or SpikeData instance
        Data array, if input was a 2d numpy array an extra axis is appended
        to the end.
    is_spikes : bool
        if True, data is in spiking format.
        if False, data is a 3d dense array.
    """

    # Data provided as a dense numpy array.
    if isinstance(data, np.ndarray) and data.ndim == 3:
        return data, False

    # Dense numpy array with 2-dimensions is okay. In this case, add an extra
    # dimension/axis (single-neuron dataset).
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        return data[:, :, None], False

    # Spiking data format
    elif isinstance(data, SpikeData):
        return data, True

    # Data format not recognized
    else:
        raise ValueError("Data input should be formatted as a 3d numpy "
                         "array (trials x times x neurons) or as a "
                         "SpikeData instance.")


def _diff_gramian(T, smoothness_scale, l2_scale):
    """Constructs regularization gramian in least-squares problem.
    """
    DtD = np.ones((3, T))

    DtD[-1] = 6.0
    DtD[-1, 1] = 5.0
    DtD[-1, 0] = 1.0
    DtD[-1, -1] = 1.0
    DtD[-1, -2] = 5.0

    DtD[-2] = -4.0
    DtD[-2, 1] = -2.0
    DtD[-2, -1] = -2.0

    DtD *= smoothness_scale
    DtD[-1] += l2_scale

    return DtD


def upsample(signal, factor, axis=-1):
    """Upsamples numpy array by linear interpolation.
    """
    signal = np.asarray(signal)
    n = signal.shape[axis]
    f = interp1d(np.linspace(0, 1, n), signal, axis=axis)
    return f(np.linspace(0, 1, int(n * factor)))
