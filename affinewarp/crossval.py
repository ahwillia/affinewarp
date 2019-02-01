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
        outfile=None, **fit_kw):
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
    **fit_kw : dict
        Additional keyword arguments are passed to model.fit(...)

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

    _valid_scoring = ('neg_mse', 'r_squared')
    if scoring not in _valid_scoring:
        raise ValueError(
            "Expected 'scoring' parameter to be one of "
            "{}.".format(', '.join(_valid_scoring))
        )

    # Get scoring function.
    score_fn = getattr(metrics, scoring)

    # Dataset dimensions.
    n_neurons = binned.shape[-1]
    n_bins = binned.shape[1]

    # Enumerate all parameter settings for each model.
    knots = np.random.randint(knot_range[0], knot_range[1], size=n_samples)
    smoothness = 10 ** np.random.uniform(*np.log10(smoothness_range),
                                         size=n_samples)
    warp_reg = 10 ** np.random.uniform(*np.log10(warpreg_range),
                                       size=n_samples)

    # Allocate space for results
    scores = np.full((n_samples, n_neurons), np.nan)
    scores = np.full((n_samples, n_neurons), np.nan)
    scores = np.full((n_samples, n_neurons), np.nan)

    fit_kw.setdefault('iterations', 50)
    loss_hists = np.empty((n_samples, n_folds, fit_kw['iterations'] + 1))

    # Set up indexing for train/test splits.
    neuron_indices = np.arange(n_neurons)

    # Allocate dictionaries that store the best models and scores.
    best_models = {k: None for k in range(*knot_range)}
    best_scores = {k: -np.inf for k in range(*knot_range)}

    # Fit models.
    for i, k, s, w in zip(trange(n_samples), knots, smoothness, warp_reg):

        # Construct model object.
        if k == -1:
            model = ShiftWarping(smoothness_reg_scale=s, warp_reg_scale=w)
        else:
            model = PiecewiseWarping(
                n_knots=k, smoothness_reg_scale=s, warp_reg_scale=w)

        # Shuffle neuron order for train and test sets.
        np.random.shuffle(neuron_indices)

        # Iterate over test sets.
        for f, testset in enumerate(np.array_split(neuron_indices, n_folds)):

            # Get indices for train set.
            testset.sort()  # needed for SpikeData selection.
            trainset = np.ones_like(neuron_indices, bool)
            trainset[testset] = False

            # Fit model to training set.
            model.fit(binned[:, :, trainset], verbose=False, **fit_kw)
            loss_hists[i, f] = model.loss_hist

            # Apply inverse warping functions to the test set.
            if data is None:
                testdata = binned[:, :, testset]
            else:
                testdata = data.select_neurons(testset)

            # Evaluate metrics.
            aligned_data = model.transform(testdata)
            neg_mse[i, testset] = metrics.neg_mse(aligned_data, n_bins)
            r_squared[i, testset] = metrics.r_squared(aligned_data, n_bins)
            snr[i, testset] = metrics.snr(aligned_data, n_bins)

        # Store best model in each knot category.
        mean_score = np.mean(neg_mse[i])
        if mean_score > best_scores[k]:
            best_scores[k] = mean_score
            best_models[k] = deepcopy(model)

        # Save results
        results = {
            'neg_mse': neg_mse[:(i+1)],
            'r_squared': r_squared[:(i+1)],
            'snr': snr[:(i+1)],
            'knots': knots[:(i+1)],
            'smoothness': smoothness[:(i+1)],
            'warp_reg': warp_reg[:(i+1)],
            'loss_hists': loss_hists[:(i+1)],
        }
        if outfile is not None:
            dd.io.save(outfile, results)

    return results, best_models


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
