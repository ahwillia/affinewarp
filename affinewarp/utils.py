import numba
import numpy as np
from .spikedata import SpikeData
from scipy.interpolate import interp1d


def check_dimensions(model, data):
    """
    Check if data dimensions match model fit.

    Parameters
    ----
    model : ShiftWarping or PiecewiseWarping instance
        Model instance.
    data : ndarray or SpikeData instance
        Input data array.

    Returns
    -------
    data : ndarray or SpikeData instance
        If data was a 2-dimensional array (matrix), an extra dimension is
        appended to the end.
    is_spikes : bool
        True if data is a SpikeData instance, False otherwise.
    """

    # Check type of data.
    if isinstance(data, SpikeData):
        K = data.n_trials

    elif isinstance(data, np.ndarray) and data.ndim == 2:
        K, T = data.shape
        data = data[:, :, None]

    elif isinstance(data, np.ndarray) and data.ndim == 3:
        K, T, _ = data.shape
    else:
        raise ValueError("Expected 'data' to be a SpikeData instance or a "
                         "numpy.ndarray with dimensions "
                         "(trials x timebins x units).")

    # Check number of trials match.
    try:
        K_model = len(model.shifts)
    except AttributeError:
        K_model = len(model.x_knots)

    if K_model != K:
        raise ValueError("Dimension mismatch: model was fit on a dataset with "
                         "{} trials, but input had {} "
                         "trials".format(K_model, K))

    # Return dataset.
    return data, isinstance(data, SpikeData)


def check_dense_data(data):

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
