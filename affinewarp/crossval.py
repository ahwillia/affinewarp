"""Validation methods for time warping models."""

import numpy as np
from tqdm import tqdm, trange
from copy import deepcopy
from .spikedata import SpikeData
from .utils import upsample
from .piecewisewarp import PiecewiseWarping
from .shiftwarp import ShiftWarping
from . import metrics
import deepdish as dd


def paramsearch(
        binned, n_samples, data=None, n_folds=5, knot_range=(-1, 2),
        smoothness_range=(1e-2, 1e2), warpreg_range=(1e-2, 1e1),
        iter_range=(50, 300), warp_iter_range=(50, 300), outfile=None):
    """
    Performs randomized search over hyperparameters on warping
    functions. For each set of randomly sampled parameters, neurons
    are randomly split `n_folds` times into train/test groups. An
    R-squared metric of across-trial reliability is measured on each
    test set; larger scores indicate warping functions that generalize
    better.

    Parameters
    ----------
    binned : ndarray
        trials x timepoints x neurons binned spikes
    n_samples : int
        Number of parameter settings to try per fold.
    data : SpikeData
        Holds unbinned spike times.
    n_folds : int
        Number of folds used for cross-validation.
    knot_range : tuple of ints
        Specifies [minimum, maximum) number of knots in warping
        functions. Uniform random integers over this includive interval
        are sampled for each model. A value of -1 denotes a shift-only
        warping model; a value of 0 denotes a linear warping model (no
        interior knots); etc.
    smoothness_range : tuple of floats
        Specifies [minimum, maximum) strength of regularization on
        template smoothness; larger values penalize roughness over time
        more stringently. The regularization strength for each model
        is randomly sampled from a log-uniform distribution over this
        interval.
    warpreg_range : tuple of floats
        Specifies [minimum, maximum) strength of regularization on the
        area between the warping functions and the identity line;
        larger values penalize warping more stringently. The
        regularization strength for each model is randomly sampled from
        a log-uniform distribution over this interval.
    iter_range : tuple of ints
        Specifies [minimum, maximum) number of iterations used to optimize
        each model, which are sampled log-uniformly over this interval
        and constrained to be integer-valued.
    warp_iter_range : tuple of ints
        Specifies [minimum, maximum) number of inner iterations to apply
        to update the warping functions on each step of optimization.
        These are also randomly sampled log-uniformly over the specified
        interval.

    outfile : None or str (optional)
        If provided, data are saved after each iteration to this filename.

    Returns
    -------
    results : dict
        Dictionary holding sampled model parameters and scores. Key-value
        pairs are:

        "neg_mse" : (n_samples x n_neurons) array holding negative mean-squared
        error score for each neuron.

        "r_squared" : (n_samples x n_neurons) array holding R-squared score
        for each each neuron.

        "snr" : (n_samples x n_neurons) array holding signal-to-noise ratio
        score for each each neuron.

        "knots" : (n_samples,) array holding number of knots in piecewise
        linear warping function for each evaluated model.

        "smoothness" : (n_samples,) array holding sampled regularization
        strengths on warping templates, penalizing roughness.

        "warp_reg" : (n_samples,) array holding sampled regularization
            strengths on warping function distance from identity.
        "loss_hists" : (n_samples, n_folds, n_iterations + 1) array
            holding the learning curves for all models.

    best_models : dict
        Dictionary mapping number of knots (int) to a ShiftWarping or
        PiecewiseWarping model instance.
    """

    # Check inputs.
    if (data is not None) and (data.n_neurons != binned.shape[-1]):
        raise ValueError(
            "Expected binned spikes and SpikeData object to have the same "
            "number of neurons."
        )

    # Dataset dimensions.
    n_trials = binned.shape[0]
    n_neurons = binned.shape[-1]
    n_bins = binned.shape[1]

    # Enumerate all parameter settings for each model.
    knots = np.random.randint(*knot_range, size=n_samples)
    smoothness = 10 ** np.random.uniform(*np.log10(smoothness_range),
                                         size=n_samples)
    warp_reg = 10 ** np.random.uniform(*np.log10(warpreg_range),
                                       size=n_samples)
    iterations = 10 ** np.random.uniform(*np.log10(iter_range),
                                         size=n_samples)
    warp_iterations = 10 ** np.random.uniform(*np.log10(warp_iter_range),
                                              size=n_samples)

    # Convert sampled iterations to integers.
    iterations = iterations.astype('int')
    warp_iterations = warp_iterations.astype('int')

    # Allocate space for training and testing loss.
    train_loss = np.full((n_samples, n_folds, n_neurons), np.nan)
    test_loss = np.full((n_samples, n_neurons), np.nan)
    loss_hists = np.full(
        (n_samples, n_folds, iter_range[1] + 1), np.nan)

    # Set up indexing for train/test splits.
    neuron_indices = np.arange(n_neurons)
    trial_indices = np.arange(n_trials)

    # Fit models.
    params = knots, smoothness, warp_reg, iterations, warp_iterations
    for i, k, s, w, itr, w_itr in zip(trange(n_samples), *params):

        # Construct model object.
        if k == -1:
            model = ShiftWarping(smoothness_reg_scale=s, warp_reg_scale=w)
        else:
            model = PiecewiseWarping(
                n_knots=k, smoothness_reg_scale=s, warp_reg_scale=w)

        # Shuffle neuron order for train and test sets.
        np.random.shuffle(neuron_indices)
        np.random.shuffle(trial_indices)

        # Form data partitions.
        neuron_splits = np.array_split(neuron_indices, n_folds)
        trial_splits = np.array_split(trial_indices, n_folds)
        splits = (neuron_splits, trial_splits)

        # Iterate over test sets.
        for f, (test_neurons, test_trials) in enumerate(zip(*splits)):

            # Get indices for train set.
            test_neurons.sort()  # needed for SpikeData selection.
            train_neurons = np.ones_like(neuron_indices, bool)
            train_neurons[test_neurons] = False
            train_trials = np.ones_like(trial_indices, bool)
            train_trials[test_trials] = True

            # Fit model to training set.
            fit_kw = {
                "verbose": False,
                "iterations": itr,
                "warp_iterations": w_itr,
                "neuron_idx": train_neurons,
                "trial_idx": train_trials,
            }
            model.fit(binned, **fit_kw)
            pred = model.predict()

            # Save learning curve
            loss_hists[i, f, :(itr+1)] = model.loss_hist

            # Save loss on intersection of training neurons and trials.
            train_pred = pred[train_trials][:, :, train_neurons]
            train_data = binned[train_trials][:, :, train_neurons]
            resid = train_pred - train_data
            train_loss[i, f, train_neurons] = \
                np.sqrt(np.mean(resid ** 2, axis=(0, 1)))

            # Save loss on test set.
            test_pred = pred[test_trials][:, :, test_neurons]
            test_data = binned[test_trials][:, :, test_neurons]
            resid = test_pred - test_data
            test_loss[i, test_neurons] = \
                np.sqrt(np.mean(resid ** 2, axis=(0, 1)))

        # Save results
        results = {
            'train_loss': np.nanmean(train_loss[:(i+1)], axis=1),
            'test_loss': test_loss[:(i+1)],
            'knots': knots[:(i+1)],
            'smoothness': smoothness[:(i+1)],
            'warp_reg': warp_reg[:(i+1)],
            'iterations': iterations[:(i+1)],
            'warp_iterations': warp_iterations[:(i+1)],
            'loss_hists': loss_hists[:(i+1)],
        }
        if outfile is not None:
            dd.io.save(outfile, results)

    return results


def heldout_transform(model, binned, data, transformed_neurons=None,
                      progress_bar=True, **fit_kw):
    """
    Transform each neuron's activity by holding it out of model fitting
    and applying warping functions fit to the remaining neurons.

    Parameters
    ----------
    models : ShiftWarping or AffineWarping instance
        Model to fit
    binned : numpy.ndarray
        Array holding binned spike times (trials x num_timebins x
        neurons)
    data : SpikeData instance
        Raw spike times.
    transformed_neurons (optional) : array-like or ``None``
        Indices of neurons that are transformed. If None, all neurons
        are transformed.
    fit_kw (optional) : dict
        Additional keyword arguments are passed to ``model.fit(...)``.

    Returns
    -------
    aligned_data : SpikeData instance
        Transformed version of ``data`` where each neuron/unit is
        independently aligned.

    Raises
    ------
    ValueError: If ``binned`` and ``data`` have inconsistent dimensions.

    Notes
    -----
    Since a different model is fit for each neuron, the warping
    functions are not necessarily consistent across neurons in the
    returned data array. Thus, each neuron should be considered as
    having its own time axis.
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
    Generate Poisson random spiking data with identical trial-average statistics.

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
