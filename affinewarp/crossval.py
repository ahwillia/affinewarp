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


def paramsearch(
        binned, n_models, data=None, heldout_frac=.2, knot_range=(-1, 1),
        smoothness_range=(1e-2, 1e2), warpreg_range=(1e-2, 1e1), **fit_kw):
    """
    Performs randomized search over hyperparameters. An R-squared metric of
    across-trial reliability is measured on a test set of neurons; larger
    scores indicate warping functions that generalize better.

    Parameters
    ----------
    binned : ndarray
        trials x timepoints x neurons binned spikes
    n_models : int
        Number of parameter settings to try.
    data : SpikeData
        Holds unbinned spike times (optional).
    heldout_frac : float
        Fraction of neurons to holdout for testing.
    knot_range : tuple of ints
        Specifies number of knots in piecewise warping functions.
        Optional, default is (-1, 2).
    smoothness_range : tuple of floats

    warpreg_range

    Returns
    -------
    rmse : (knots x smoothness x warp regularization)
    knots
    smoothness
    warp_reg
    """

    # Check inputs.
    if (data is not None) and (data.n_neurons != binned.shape[-1]):
        raise ValueError(
            "Expected binned spikes and SpikeData object to have the same "
            "number of neurons."
        )

    # Dataset dimensions.
    n_neurons = binned.shape[-1]
    n_bins = binned.shape[1]
    n_test = int(n_neurons * heldout_frac)

    # Enumerate all parameter settings for each model.
    knots = np.random.randint(knot_range[0], knot_range[1] + 1, size=n_models)
    smoothness = 10 ** np.random.uniform(*np.log10(smoothness_range),
                                         size=n_models)
    warp_reg = 10 ** np.random.uniform(*np.log10(warpreg_range),
                                       size=n_models)

    # Define train set and test set partitions.
    neurons = np.arange(n_neurons)
    testset = np.sort(np.random.choice(neurons, n_test, replace=False))
    if data is None:
        testdata = binned[:, :, testset]
    else:
        testdata = data.select_neurons(testset)
    trainset = np.setdiff1d(neurons, testset)
    traindata = binned[:, :, trainset]

    # Compute scores without warping.
    raw_scores = metrics.r_squared(testdata, n_bins)

    # Allocate space for results
    scores = np.empty((n_models, len(testset)))

    # Fit models.
    for i, k, s, w in zip(trange(n_models), knots, smoothness, warp_reg):

        # Construct model object
        if k == -1:
            model = ShiftWarping(smoothness_reg_scale=s, warp_reg_scale=w)
        else:
            model = PiecewiseWarping(n_knots=k, smoothness_reg_scale=s, warp_reg_scale=w)

        # Fit model to training set.
        model.fit(traindata, verbose=False, **fit_kw)

        # Compute scores on test set.
        scores[i] = metrics.rmse(model.transform(testdata), n_bins)

    return scores, raw_scores, knots, smoothness, warp_reg


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
