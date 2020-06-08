"""Experimental code on Multi-Warping."""

import numpy as np
from .shiftwarp import ShiftWarping
from tqdm import trange


class MultiShiftWarping:
    """
    Extension of shift-only warping to uncover multiple latent events.
    """

    def __init__(self, n_templates, **model_args):

        # Check that Poisson loss is not specified.
        model_args.setdefault('loss', 'quadratic')
        if model_args['loss'] == 'poisson':
            raise ValueError('MultiWarp only supports quadratic loss.')

        # Initialize models.
        self.models = [ShiftWarping(**model_args) for m in range(n_templates)]

    def fit(self, data, iterations=20, verbose=True):

        data = data.astype('float')
        n_trials, n_time, n_units = data.shape

        # Initialize templates and warps manually.
        pred = np.zeros_like(data)
        for model in self.models:
            L = model.maxlag * n_time
            model.shifts = np.random.randint(-L, L, size=n_trials)
            model.template = None
            model._fit_template(data - pred, model.shifts)
            pred += model.predict()

        # # Initialize templates and warps manually.
        # bases = _mspline_basis(n_time, len(self.models))
        # psth = data.mean(axis=0)
        # for k in range(len(self.models)):
        #     self.models[k].shifts = np.zeros(n_trials, dtype=int)
        #     self.models[k].template = 0.2 * psth
        #     self.models[k].template += 0.8 * bases[k][:, None] * psth
        #     psth * bases[k][:, None]

        # # Compute total model prediction.
        # pred = self.predict()

        # Fit each model to its residual
        self.loss_hist = [np.mean((self.predict() - data)**2)]
        pbar = trange(iterations) if verbose else range(iterations)
        for i in pbar:
            for k, model in enumerate(self.models):
                model.fit(
                    data - self.predict(skip=[k]),
                    iterations=2, verbose=False)
            self.loss_hist.append(
                np.mean((self.predict() - data)**2))

    def predict(self, skip=[]):
        K = len(self.models[0].shifts)
        T, N = self.models[0].template.shape
        pred = np.zeros((K, T, N))
        for k, m in enumerate(self.models):
            if k not in skip:
                pred += self.models[k].predict()
        return pred

    def transform(self, data, index):
        return self.models[index].transform(data)

    def partition_spikes(self, data):
        """
        Attributes each spike to one of the latent warping models.

        Parameters
        ----------
        data : SpikeData
            Full dataset.

        Returns
        -------
        partitions : list of SpikeData instances
            List holding spikes for each latent process.
        """

        # Number of time bins.
        num_bins = model_idx.shape[1]

        # Convert spiketimes to bin ids.
        bin_ids = (self.fractional_spiketimes * (n_bins - 1e-9)).astype(int)

        # For each time bin, trial, and neuron, determine which model has highest
        # predicted firing rate.
        model_idx = np.argmax([m.predict() for m in self.models], axis=0)

        # Create one new SpikeData object for each warping model.
        trials = [[] for m in self.models]
        spiketimes = [[] for m in self.models]
        neurons = [[] for m in self.models]

        # Attribute each spike to one of the latent models.
        for k, b, t, n in zip(data.trials, bin_ids, data.spiketimes, data.neurons):
            i = model_idx[k, b, n]
            trials[i].append(k)
            spiketimes[i].append(t)
            neurons[i].append(n)

        # Create spike data objects
        t0, t1 = data.tmin, data.tmax
        dims = dict(n_trials=data.n_trials, n_neurons=data.n_neurons)
        for kk, tt, nn in zip(trials, times, neurons):
            partitions.append(SpikeData(kk, tt, nn, t0, t1, **dims))

        return partitions


# ====================== #
# == Helper functions == #
# ====================== #


def _mspline_basis(n_bins, n_models):
    """
    Generates a set of M-spline basis functions.
    """
    x = np.linspace(0, 1, n_bins)
    order = n_models - 1
    n_spline_knots = n_models - order + 2
    centers = np.linspace(0, 1, n_spline_knots)
    bases = []
    for i in range(1 - order, len(centers) - 1):
        bases.append(_mspline(x, centers, order, i))
    return bases


def _mspline(x, centers, order, idx):
    """
    Generates an M-spline function.

    Parameters
    ----------
    x : ndarray
        vector specifying x-locations
    centers : ndarray
        vector specifying x-locations of knots
    order : int
        spline order (large numbers result in smoother splines)
    idx : int
        specifies which spline to compute

    Returns
    -------
    y : ndarray
        vector specifying spline values at each x-location
    """

    i0 = np.clip(idx, 0, len(centers)-1)
    i1 = np.clip(idx + order, 0, len(centers)-1)

    if centers[i1] - centers[i0] == 0:
        return 0.0 * x

    if order == 1:
        in_support = (centers[i0] <= x) & (x < centers[i1])
        return in_support / (centers[i1] - centers[i0])

    num = (x - centers[i0]) * _mspline(x, centers, order - 1, idx) + \
        (centers[i1] - x) * _mspline(x, centers, order - 1, idx + 1)

    denom = (order - 1) * (centers[i1] - centers[i0])

    return order * num / denom
