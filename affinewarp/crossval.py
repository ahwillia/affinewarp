"""Validation methods for time warping models."""

import itertools
import numpy as np
import numpy.random as npr
from tqdm import tqdm, trange
from copy import deepcopy
from .spikedata import SpikeData
from .utils import upsample
from .piecewisewarp import PiecewiseWarping
from .shiftwarp import ShiftWarping
from . import metrics
from ._optimizers import nowarp_template
EPS = np.finfo(float).eps


def _crossval_partition(N, train_folds, valid_folds, test_folds):
    """
    Computes a randomized train-validation-test partition of N indices.
    """
    num_folds = train_folds + valid_folds + test_folds
    splits = np.array_split(npr.permutation(N), num_folds)
    splits = [splits[i] for i in npr.permutation(num_folds)]
    trainset = np.concatenate(splits[:train_folds])
    validset = np.concatenate(splits[train_folds:(train_folds + valid_folds)])
    testset = np.concatenate(splits[(train_folds + valid_folds):])
    return trainset, validset, testset


def _sample_log_uniform(rng, size):
    """
    Samples from a log-uniform distribution.
    """
    return 10 ** np.random.uniform(*np.log10(rng), size=size)


def _crossval_loss(pred, targ, kk, nn):
    """
    Computes norm of residuals relative to norm of data, for a subset of
    trials (indexed by kk) and units (indexed by nn).
    """
    resid = pred[kk][:, :, nn] - targ[kk][:, :, nn]
    num = np.linalg.norm(resid)
    denom = np.linalg.norm(targ[kk][:, :, nn]) + EPS
    return num / denom


def baseline_performance(
        binned, n_samples, n_valid_samples, n_train_folds=3,
        n_valid_folds=1, n_test_folds=1, smoothness_range=(1e-2, 1e2)):
    """
    Performs randomized nested cross-validation over a range of
    template smoothness regularization scales, assuming no warping
    is done.

    Parameters
    ----------
    binned : ndarray
        trials x timepoints x neurons binned spikes
    samples_per_knot : int
        Number of cross-validation runs per knot.
    n_valid_samples : int
        Number of inner samples to optimize smoothness and warp
        complexity regularization parameters on validation set.
    n_train_folds : int
        Number of folds used for training.
    n_valid_folds : int
        Number of folds used for validation.
    n_test_folds : int
        Number of folds used for testing.
    smoothness_range : tuple of floats
        Specifies [minimum, maximum) strength of regularization on
        template smoothness; larger values penalize roughness over time
        more stringently. The regularization strength for each model
        is randomly sampled from a log-uniform distribution over this
        interval.

    Returns
    -------
    results : dict
        Dictionary holding results:

        "smoothness" : (n_samples, n_valid_samples) array holding sampled
            regularization strengths on warping templates, penalizing
            roughness.

        "train_loss": (n_samples, n_valid_samples) array holding model loss
            on the training set.

        "test_loss": (n_samples,) array holding model loss on the test set.


    Notes
    -----
    Only implemented for quadratic loss.
    """

    # Data dimensions.
    K, T, N = binned.shape

    # Use default L2 regularization for PiecewiseWarping.
    l2_scale = 1e-7

    # Grid search over logarithmic scale.
    smoothness = _sample_log_uniform(
        smoothness_range, size=(n_samples, n_valid_samples))

    # Initialize arrays to store losses.
    train_loss = np.empty((n_samples, n_valid_samples))
    valid_loss = np.full((n_samples, n_valid_samples), np.inf)
    test_loss = np.empty(n_samples)

    progress_bar = tqdm(total=n_samples * n_valid_samples)

    for i, j in itertools.product(range(n_samples), range(n_valid_samples)):

        # Update train - validation - test sets.
        if j == 0:
            train_trials, val_trials, test_trials = _crossval_partition(
                binned.shape[0], n_train_folds, n_valid_folds, n_test_folds)

        # Fit template and form prediction.
        s = smoothness[i, j]
        template = nowarp_template(binned[train_trials], s, l2_scale)
        pred = np.tile(template[None, :, :], (K, 1, 1))

        # Record loss on training set.
        train_loss[i, j] = _crossval_loss(
                pred, binned, train_trials, slice(None))
        valid_loss[i, j] = _crossval_loss(
            pred, binned, val_trials, slice(None))

        # Save loss on test set if validation loss is optimal
        if np.argmin(valid_loss[i]) == j:
            test_loss[i] = _crossval_loss(
                pred, binned, test_trials, slice(None))

        # Update progress bar.
        progress_bar.update(1)

    return {
        "smoothness": smoothness,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "test_loss": test_loss,
    }


def paramsearch(
        binned, samples_per_knot, n_valid_samples, n_train_folds=3,
        n_valid_folds=1, n_test_folds=1, knot_range=(-1, 2),
        smoothness_range=(1e-2, 1e2), warpreg_range=(1e-2, 1e1),
        iter_range=(50, 300), warp_iter_range=(50, 300), outfile=None):
    """
    Performs nested cross-validation over shift-only, linear, and
    piecewise linear warping models, in order to tune all hyperparmeters
    and compare performance. For each set of randomly sampled parameters,
    trials and units are randomly split `n_folds` times into train/test
    groups. An R-squared metric of across-trial reliability is measured
    on each test set; larger scores indicate warping functions that generalize
    better.

    Parameters
    ----------
    binned : ndarray
        trials x timepoints x neurons binned spikes
    samples_per_knot : int
        Number of cross-validation runs per knot.
    n_valid_samples : int
        Number of inner samples to optimize smoothness and warp
        complexity regularization parameters on validation set.
    n_train_folds : int
        Number of folds used for training.
    n_valid_folds : int
        Number of folds used for validation.
    n_test_folds : int
        Number of folds used for testing.
    knot_range : tuple of ints
        Specifies [minimum, maximum) number of knots in warping
        functions. A value of -1 denotes a shift-only
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
        Dictionary holding results:

        "knots" : (n_samples,) array holding number of knots in piecewise
            linear warping function for each evaluated model.

        "smoothness" : (n_samples, n_valid_samples) array holding sampled
            regularization strengths on warping templates, penalizing
            roughness.

        "warp_reg" : (n_samples, n_valid_samples) array holding sampled
            regularization strengths on warping function distance from
            identity.

        "iterations" : (n_samples, n_valid_samples) array holding number
            of model optimization steps.

        "warp_iterations" : (n_samples, n_valid_samples) array holding number
            of inner iteration steps for fitting warping functions.

        "train_loss": (n_samples, n_valid_samples) array holding model loss
            on the training set.

        "valid_loss": (n_samples, n_valid_samples) array holding model loss
            on the validation set.

        "test_loss": (n_samples,) array holding model loss on the test set.

        "loss_hists" : (n_samples, n_valid_samples, n_iterations + 1) array
            holding the learning curves for all models. The loss is computed
            over the combined train and validation set.

    Notes
    -----
    Only implemented for quadratic loss.
    """

    # Dataset dimensions (trials x timepoints x units).
    K, T, N = binned.shape

    # Randomly draw all parameter settings for each model.
    knots = np.tile(np.arange(*knot_range), samples_per_knot)
    n_samples = len(knots)

    smoothness = lg_unif(
        smoothness_range, size=(n_samples, n_valid_samples))
    warp_reg = lg_unif(
        warpreg_range, size=(n_samples, n_valid_samples))
    iterations = lg_unif(
        iter_range, size=(n_samples, n_valid_samples)).astype('int')
    warp_iterations = lg_unif(
        warp_iter_range, size=(n_samples, n_valid_samples)).astype('int')

    # Initialize arrays to store losses.
    train_loss = np.empty((n_samples, n_valid_samples))
    valid_loss = np.full((n_samples, n_valid_samples), np.inf)
    test_loss = np.empty(n_samples)
    loss_hists = np.full(
        (n_samples, n_valid_samples, iter_range[1] + 1), np.nan)

    progress_bar = tqdm(total=n_samples * n_valid_samples)

    for i, j in itertools.product(range(n_samples), range(n_valid_samples)):

        # Update train - validation - test sets.
        if j == 0:
            train_units, val_units, test_units = _crossval_partition(
                N, n_train_folds, n_valid_folds, n_test_folds)
            train_trials, val_trials, test_trials = _crossval_partition(
                K, n_train_folds, n_valid_folds, n_test_folds)

        # Create model instance.
        model_kw = {
            "smoothness_reg_scale": smoothness[i, j],
            "warp_reg_scale": warp_reg[i, j]
        }
        if knots[i] == -1:
            model = ShiftWarping(**model_kw)
        else:
            model = PiecewiseWarping(n_knots=knots[i], **model_kw)

        # Fit model.
        fit_kw = {
            "verbose": False,
            "iterations": iterations[i, j],
            "warp_iterations": warp_iterations[i, j],
            "neuron_idx": train_units,
            "trial_idx": train_trials,
        }
        model.fit(binned, **fit_kw)

        loss_hists[i, j, :(iterations[i, j] + 1)] = model.loss_hist

        # Record loss on training set.
        pred = model.predict()
        train_loss[i, j] = _crossval_loss(
                pred, binned, train_trials, train_units)
        valid_loss[i, j] = _crossval_loss(
            pred, binned, val_trials, val_units)

        # Save loss on test set if validation loss is optimal
        if np.argmin(valid_loss[i]) == j:
            test_loss[i] = _crossval_loss(
                pred, binned, test_trials, test_units)

        # Save results.
        if j == n_valid_samples - 1:
            results = {
                "knots": knots[:(i+1)],
                "smoothness": smoothness[:(i+1)],
                "warp_reg": warp_reg[:(i+1)],
                "iterations": iterations[:(i+1)],
                "warp_iterations": warp_iterations[:(i+1)],
                "train_loss": train_loss[:(i+1)],
                "valid_loss": valid_loss[:(i+1)],
                "test_loss": test_loss[:(i+1)],
                "loss_hists": loss_hists[:(i+1)],
            }
            if outfile is not None:
                np.savez(outfile, **results)

        # Update progress bar.
        progress_bar.update(1)

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
