"""Validation methods for time warping models."""

import numpy as np
from tqdm import tqdm
import sparse
from copy import deepcopy
from .spikedata import SpikeData
from .utils import upsample

# TODO(ahwillia)
# --------------
# Add K-fold cross-validation function to determine all hyperparameters. Need
# to hold out trials to fit regularization on trial-averaging procedure. Need
# to hold out neurons to fit warping hyperparameters.


def heldout_transform(model, binned, data, transformed_neurons=None, **fit_kw):
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

    # Hold out each neuron, fit models, and apply transform to heldout cell.
    for n in tqdm(transformed_neurons):

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
