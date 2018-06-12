import numpy as np
from tqdm import trange
import sparse
from copy import deepcopy
from .spikedata import bin_spikes
from scipy.interpolate import interp1d


def kfold(N, n_splits):
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

    # transformed spike times
    aligned_data = [[] for m in models]

    # hold out each feature, and compute its transforms
    for n in trange(n_neurons):

        # define training set
        trainset = list(set(range(n_neurons)) - {n})

        # fit model and save transformed test set
        for i, m in enumerate(models):
            m.fit(binned[:, :, trainset], **fit_kw[m])

            # warp test set
            aligned_data[i].append(m.transform(data[:, :, n]))

    # concatenate transformed data
    if isinstance(data, sparse.COO):
        aligned_data = [sparse.concatenate(a, axis=2) for a in aligned_data]
    else:
        aligned_data = [np.concatenate(a, axis=2) for a in aligned_data]

    # squeeze results if a single model was provided
    if len(aligned_data) == 1:
        aligned_data = aligned_data[0]

    return aligned_data


def null_dataset(data, nbins):
    """
    Generate Poisson random spiking pattern on each trial.
    """

    # num trials, num timepoints, num neurons
    K, T, N = data.shape

    # trial-average estimate of firing rates
    binlen = T / nbins  # length of each time bin
    psth = bin_spikes(data, nbins).mean(axis=0) / binlen  # spike rate

    # interpolate binned firing rates to length of spike data
    interp_func = interp1d(np.arange(nbins), psth, axis=0)
    psth_interp = interp_func(np.linspace(0, nbins-1, T))

    # draw poisson random data and package into sparse array
    return sparse.COO(np.random.poisson(psth_interp, size=(K, T, N)))
