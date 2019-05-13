"""Common metrics for evaluating across-trial variability."""

import numpy as np
from .spikedata import SpikeData
from .piecewisewarp import PiecewiseWarping
from .shiftwarp import ShiftWarping
from sklearn.utils.validation import check_is_fitted
from numba import jit


def mse(data, nbins=None):
    """
    Mean-Squared-Error of trial-average for each neuron.
    """
    binned = data.bin_spikes(nbins) if isinstance(data, SpikeData) else data
    resid = binned - binned.mean(axis=0, keepdims=True)
    return np.mean(resid ** 2, axis=(0, 1))


def rmse(data, nbins=None):
    """
    Root-Mean-Squared-Error of trial-average for each neuron.
    """
    return np.sqrt(mse(data, nbins=nbins))


def neg_mse(data, nbins=None):
    """
    Negative Mean-Squared-Error of trial-average for each neuron.
    """
    return -mse(data, nbins=nbins)


def r_squared(data, nbins=None):
    """
    Coefficient of determination of trial-average for each neuron.

    Parameters
    ----------
    data : ndarray or SpikeData
        Either a ndarray holding binned spike times with shape
        (trials x timepoints x neurons). Or a SpikeData instance holding
        unbinned spike times.
    nbins : int or None
        Number of bins. Ignored if data are already binned.

    Returns
    -------
    scores : ndarray
        Computed R-squared value for each neuron.
    """
    binned = data.bin_spikes(nbins) if isinstance(data, SpikeData) else data
    # constant firing rate model
    resid = binned - binned.mean(axis=(0, 1), keepdims=True)
    ss_data = np.sum(resid ** 2, axis=(0, 1))
    # psth model
    resid = binned - binned.mean(axis=0, keepdims=True)
    ss_model = np.sum(resid ** 2, axis=(0, 1))
    # return explained variance
    return 1 - ss_model / ss_data


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
        binned = data.bin_spikes(nbins)
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


def warp_distances(model_1, model_2):
    """
    Computes distance between the warping functions of two models.

    Parameters
    ----------
    model_1 : ``PiecewiseWarping`` or ``ShiftWarping`` instance.
        Fitted model instance.
    model_2 : ``PiecewiseWarping`` or ``ShiftWarping`` instance.
        Another fitted model instance.

    Returns
    -------
    distances : ndarray
        Vector of length ``n_trials``

    Raises
    ------
    ValueError
        If models are not fit (undefined warping functions) or were fit on
        datasets with inconsistent numbers of trials.
    """

    # Check inputs.
    model_1.assert_fitted()
    model_2.assert_fitted()

    # If ShiftWarping
    if isinstance(model_1, ShiftWarping) and isinstance(model_2, ShiftWarping):
        return np.abs(model_1.fractional_shifts - model_2.fractional_shifts)

    if isinstance(model_1, ShiftWarping):
        model_1 = PiecewiseWarping().copy_fit(model_1)
    if isinstance(model_2, ShiftWarping):
        model_2 = PiecewiseWarping().copy_fit(model_2)

    n_trials = len(model_1.x_knots)
    if len(model_2.x_knots) != n_trials:
        raise ValueError("Dimension mismatch. Number of trials in model_1 "
                         "is {}, but number of trials in model_2 "
                         "is {}".format(n_trials, len(model_2.x_knots)))

    new_knots = model_1.n_knots + model_2.n_knots + 2
    new_x_knots = np.full((n_trials, new_knots), np.nan)
    new_y_knots = np.full((n_trials, new_knots), np.nan)

    knots = (model_1.x_knots,
             model_1.y_knots,
             model_2.x_knots,
             model_2.y_knots,
             new_x_knots,
             new_y_knots)

    for kn in zip(*knots):
        _subtract_piecewise(*kn)

    distances = _piecewise_integral(new_x_knots, new_y_knots,
                                    np.full(n_trials, np.nan))
    return distances


@jit(nopython=True)
def _subtract_piecewise(x1, y1, x2, y2, new_x, new_y):
    """Subtract two piecewise linear functions.

    Parameters
    ----------
    x1 : ndarray
        x knots for first piecewise function.
    y1 : ndarray
        y knots for first piecewise function.
    x2 : ndarray
        x knots for second piecewise function.
    y2 : ndarray
        y knots for second piecewise function.
    new_x : ndarray
        Storage for x knots for the resulting piecewise function.
    new_y : ndarray
        Storage for y knots for the resulting piecewise function.
    """
    # initial point.
    new_x[0] = 0.0
    new_y[0] = y1[0] - y2[0]

    # initial slopes of input functions.
    m1 = (y1[1] - y1[0]) / (x1[1] - x1[0])
    m2 = (y2[1] - y2[0]) / (x2[1] - x2[0])

    # initial slope of resulting function.
    m = m1 - m2

    i, j, k = 1, 1, 1
    while (i < len(x1) - 1) or (j < len(x2) - 1):
        if x1[i] < x2[j]:
            # Add knot at x1 position.
            new_x[k] = x1[i]
            new_y[k] = new_y[k-1] + m * (new_x[k] - new_x[k-1])
            i += 1
            k += 1
            m1 = (y1[i] - y1[i-1]) / (x1[i] - x1[i-1])
            m = m1 - m2
        else:
            # Add knot at x2 position.
            new_x[k] = x2[j]
            new_y[k] = new_y[k-1] + m * (new_x[k] - new_x[k-1])
            j += 1
            k += 1
            m2 = (y2[j] - y2[j-1]) / (x2[j] - x2[j-1])
            m = m1 - m2

    # final point.
    new_x[-1] = 1.0
    new_y[-1] = y1[-1] - y2[-1]

    return new_x, new_y


@jit(nopython=True)
def _piecewise_integral(x_knots, y_knots, result):
    """Absolute value of integrated piecewise linear function."""
    for k in range(x_knots.shape[0]):
        result[k] = 0.0

        for j in range(1, x_knots.shape[1]):

            # left and right points of line segments.
            x0 = x_knots[k, j-1]
            y0 = y_knots[k, j-1]
            x1 = x_knots[k, j]
            y1 = y_knots[k, j]

            # if y0 and y1 have opposite signs (area of two triangles)
            if ((y0 < 0) and (y1 > 0)) or ((y0 > 0) and (y1 < 0)):
                v = y1 / (y1 - y0)
                result[k] += 0.5 * (x1-x0) * ((1-v)*abs(y0) + v*abs(y1))

            # if y0 and y1 have the same sign (areas of a trapezoid)
            else:
                result[k] += 0.5 * abs(y0 + y1) * (x1 - x0)

    return result
