"""Validation methods for time warping models."""

import numpy as np
from tqdm import tqdm, trange
import sparse
from copy import deepcopy
from .spikedata import SpikeData
from .utils import upsample
from .piecewisewarp import PiecewiseWarping
from .shiftwarp import ShiftWarping
from . import metrics


def paramsearch_rmse(
        binned, data, n_samples, min_knots=0, max_knots=2,
        min_smooth_scale=1e-4, max_smooth_scale=1e1, min_warp_reg=1e-4,
        max_warp_reg=1e1):
    """
    Performs grid search over hyperparameters, leaving out one neuron at a time
    and evaluating across-trial reliability on warped spike times.

    Parameters
    ----------
    binned
    data
    cv_grid_size
    min_knots
    max_knots
    min_smooth_scale
    max_smooth_scale
    min_reg_scale
    max_reg_scale

    Returns
    -------
    rmse : (knots x smoothness x warp regularization)
    knots
    smoothness
    warp_reg
    """

    # Enumerate all parameter settings for each.
    knots = np.random.randint(min_knots, max_knots + 1, size=n_samples)
    smoothness = 10 ** np.random.uniform(np.log10(min_smooth_scale),
                                         np.log10(max_smooth_scale),
                                         size=n_samples)
    warp_reg = 10 ** np.random.uniform(np.log10(min_warp_reg),
                                       np.log10(max_warp_reg),
                                       size=n_samples)

    scores = []
    for _, k, s, w in zip(trange(n_samples), knots, smoothness, warp_reg):
        # Construct model object
        if k > 0:
            model = PiecewiseWarping(n_knots=k, smoothness_reg_scale=s, warp_reg_scale=w)
        else:
            model = ShiftWarping(smoothness_reg_scale=s, warp_reg_scale=w)

        # Transform each neuron
        aligned_data = heldout_transform(model, binned, data, progress_bar=False)
        scores.append(metrics.rmse(aligned_data, nbins=binned.shape[1]))

    return np.asarray(scores), knots, smoothness, warp_reg


def heldout_transform(model, binned, data, transformed_neurons=None,
                      progress_bar=True, **fit_kw):
    """
    Transform each neuron's activity by holding it out of model fitting and
    applying warping functions fit to the remaining neurons.

    Parameters
    ----------
    models : ShiftWarping or AffineWarping instance
        Model to fit
    binned : numpy.ndarray
        Array holding binned spike times (trials x num_timebins x neurons)
    data : SpikeData instance
        Raw spike times.
    transformed_neurons (optional) : array-like or ``None``
        Indices of neurons that are transformed. If None, all neurons are
        transformed.
    fit_kw (optional) : dict
        Additional keyword arguments are passed to ``model.fit(...)``.

    Returns
    -------
    aligned_data : SpikeData instance
        Transformed version of ``data`` where each neuron/unit is independently
        aligned.

    Raises
    ------
    ValueError: If ``binned`` and ``data`` have inconsistent dimensions.

    Notes
    -----
    Since a different model is fit for each neuron, the warping functions are
    not necessarily consistent across neurons in the returned data array. Thus,
    each neuron should be considered as having its own time axis.
    """

    # broadcast keywords into dict, with model instances as keys
    fit_kw['verbose'] = False

    # data dimensions
    n_neurons = data.n_neurons
    n_trials = data.n_trials
    if (n_trials != binned.shape[0]) or (n_neurons != binned.shape[-1]):
        raise ValueError('Dimension mismatch. Binned data and spike data do '
                         'not have the same number of neurons or trials.')

    # Allocate storage for held out spike times.
    trials, spiketimes, neurons = [], [], []

    # Determine neurons to hold out and fit.
    if transformed_neurons is None:
        transformed_neurons = range(n_neurons)

    # Set up progress bar.
    if progress_bar:
        transformed_neurons = tqdm(transformed_neurons)

    # Hold out each neuron, fit models, and apply transform to heldout cell.
    for n in transformed_neurons:

        # Define training set.
        trainset = list(set(range(n_neurons)) - {n})

        # Fit model.
        model.fit(binned[:, :, trainset], **fit_kw)

        # Apply warping to test set.
        w = model.transform(data.select_neurons([n]))

        # Store result.
        trials.extend(w.trials)
        spiketimes.extend(w.spiketimes)
        neurons.extend(np.full(len(w.trials), n).tolist())

    # Package result into a SpikeData instance.
    return SpikeData(trials, spiketimes, neurons, data.tmin, data.tmax)


def null_dataset(data, nbins, upsample_factor=10):
    """
    Generate Poisson random spiking pattern on each trial.

    Parameters
    ----------
    data: SpikeData
        Spike train dataset.
    nbins: int
        Number of time bins to use when computing the trial-average PSTH.
    upsample_factor: float
        How much to upsample synthetic spiketimes over nbins.

    Returns
    -------
    null_data: SpikeData
        Poisson random spike times matching the trial-average firing rates of
        'data'.
    """

    # Trial-average estimate of firing rates.
    psth = data.bin_spikes(nbins).mean(axis=0)

    # Interpolate binned firing rates to length of spike data.
    up_psth = upsample(psth, upsample_factor, axis=0) / upsample_factor

    # Draw poisson random data.
    null_data = SpikeData([], [], [], data.tmin, data.tmax)
    for k in range(data.n_trials):
        t, neurons = np.where(np.random.poisson(up_psth))
        spiketimes = (t / up_psth.shape[0]) * (data.tmax - data.tmin) + data.tmin
        null_data.add_trial(spiketimes, neurons)

    return null_data
