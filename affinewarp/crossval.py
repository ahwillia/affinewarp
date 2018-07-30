"""Validation methods for time warping models."""

import numpy as np
from tqdm import trange
import sparse
from copy import deepcopy
from .spikedata import SpikeData
from scipy.interpolate import interp1d


def _kfold(N, n_splits):
    """Iterator for Kfold cross-validation
    """
    rng = np.random.permutation(N)

    stride = N / n_splits
    i = 0

    while i < n_splits:
        j = int(i * stride)
        k = int((i+1) * stride)
        test = rng[j:k]
        train = np.array(list(set(rng) - set(test)))
        test.sort()
        train.sort()
        yield train, test
        i += 1


# TODO(ahwillia)
# --------------
# Add K-fold cross-validation function to determine all hyperparameters. Need
# to hold out trials to fit regularization on trial-averaging procedure. Need
# to hold out neurons to fit warping hyperparameters.


def heldout_transform(models, binned, data=None, warmstart=True, **fit_kw):
    """
    Transform each neuron's activity by holding it out of model fitting and
    applying warping functions fit to the remaining neurons.

    Parameters
    ----------
    models : iterable
        sequence of models to be fit
    binned : numpy.ndarray
        array holding binned spike times (trials x times x neurons)
    data (optional) : numpy.ndarray or sparse.COO
        Array holding data to be transformed
    warmstart (optional) : bool
        If True, initialize warps with learned from last model fit.
    """

    # make models iterable
    if not np.iterable(models):
        models = (models,)

    # broadcast keywords into dict, with model instances as keys
    fit_kw['verbose'] = False
    fit_kw = {m: deepcopy(fit_kw) for m in models}

    # warmstart each model from the warps fit on the previous model.
    if warmstart:
        for m1, m0 in zip(models[1:], models):
            fit_kw[m1]['init_warps'] = m0

    # data dimensions
    n_neurons = binned.shape[-1]

    # if no data is provided, transform binned data
    if data is None:
        data = binned.copy()

    # Allocate storage for held out spike times.
    empty_spikes = SpikeData([], [], [], data.tmin, data.tmax)
    aligned_data = [empty_spikes.copy() for m in models]

    # Hold out each neuron, fit models, and apply transform to heldout cell.
    for n in trange(n_neurons):

        # Define training set.
        trainset = list(set(range(n_neurons)) - {n})

        # Fit each model.
        for i, m in enumerate(models):
            m.fit(binned[:, :, trainset], **fit_kw[m])

            # Apply warping to test set.
            w = m.transform(data.select_neurons([n]), rename_indices=False)

            # Store result on test set.
            aligned_data[i].append(w, resort_indices=False)

    # Lexographically sort spike times in each object.
    [d.sort_spikes() for d in aligned_data]

    # Squeeze results if a single model was provided.
    if len(aligned_data) == 1:
        aligned_data = aligned_data[0]

    return aligned_data


def null_dataset(data, nbins, resolution=1000):
    """
    Generate Poisson random spiking pattern on each trial.

    Parameters
    ----------
    data: SpikeData
        Spike train dataset.
    nbins: int
        Number of time bins to use when computing the trial-average PSTH.
    resolution: int
        Number of time bins to use when drawing spike times from null
        distribution.

    Returns
    -------
    null_data: SpikeData
        Poisson random spike times matching the trial-average firing rates of
        'data'.
    """

    # Trial-average estimate of firing rates.
    binlen = (data.tmax - data.tmin) / nbins  # duration of each time bin
    psth = data.bin_spiked(nbins).mean(axis=0) / binlen  # spike rate

    # Interpolate binned firing rates to length of spike data.
    f = interp1d(np.arange(nbins), psth, axis=0)
    psth_upsampled = f(np.linspace(data.tmin, data.tmax, resolution))
    psth_upsampled *= nbins / resolution

    # Draw poisson random data.
    null_data = SpikeData([], [], [], data.tmin, data.tmax)
    for k in range(data.n_trials):
        null_data.add_trial(np.random.poisson(psth_upsampled))

    return null_data
