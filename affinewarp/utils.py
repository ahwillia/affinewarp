import numpy as np
from numba import jit
import sparse
from .spikedata import is_spike_data


@jit(nopython=True)
def min_max_1d(arr):
    """
    Return maximum and minimum value of a 1d array.
    """
    vmax = arr[0]
    vmin = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > vmax:
            vmax = arr[i]
        elif arr[i] < vmin:
            vmin = arr[i]
    return vmin, vmax


def check_data_tensor(data):
    """
    Check if input is in an appropriate data format

    Returns
    -------
        arr : ndarray or spike data format
            reformatted data array
        is_spikes : bool
            if True, data is in spiking format.
            if False, data is a 3d dense array.
    """

    # add extra dimension if necessary
    if isinstance(data, (sparse.COO, np.ndarray)) and data.ndim == 2:
        data = data[:, :, None]

    # check if data is a valid spike data format
    if is_spike_data(data):
        return data, True

    # otherwise, try interpreting as a dense array
    if (
        not isinstance(data, np.ndarray) or
        data.ndim != 3 or
        not np.issubdtype(data.dtype, np.number)
    ):
        raise ValueError(
            "Data input should be formatted as a 3d numpy array (trials x "
            "times x neurons) or a spike data format. Accepted spike data "
            "formats include sparse 3d arrays (trials x times x neurons) in "
            "COO format or a tuple of three, 1d arrays holding integer "
            "indices corresponding to trial number, spike time, neuron id (in "
            "that order)."
        )

    return data, False


def _diff_gramian(T, smoothness_scale, l2_scale):
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
