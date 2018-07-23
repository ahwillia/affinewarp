"""
Specialized code for handling spike data.
"""
import itertools
import numpy as np
from numba import jit
from copy import deepcopy
from .utils import min_max_1d
from scipy.stats import rankdata


class SpikeData(object):
    """
    Represents a collection multi-dimensional spike train data.
    """
    def __init__(self, trials, spiketimes, neurons, tmin, tmax):
        """
        Parameters
        ----------
        trials : array-like
            Integer ids/indices for trial of each spike.
        spiketimes: array-like
            Float timepoints for each spike (within-trial time).
        neurons: array-like
            Integer ids/indices for neuron/electrode firing each spike.
        """

        # Treat inputs as numpy arrays.
        trials = np.asarray(trials)
        spiketimes = np.asarray(spiketimes)
        neurons = np.asarray(neurons)

        # All inputs must be 1-dimensional.
        if not (trials.ndim == spiketimes.ndim == neurons.ndim == 1):
            raise ValueError("All inputs must be 1D arrays.")

        # Trial and neuron ids must be nonnegative integers.
        if not np.issubdtype(trials.dtype, np.integer):
            raise ValueError("Trial IDs must be integers.")
        if not np.issubdtype(neurons.dtype, np.integer):
            raise ValueError("Neuron IDs must be integers.")

        min_trial, max_trial = min_max_1d(trials)
        if min_trial < 0:
            raise ValueError("Trial IDs can't be negative.")
        min_neuron, max_neuron = min_max_1d(neurons)
        if min_neuron < 0:
            raise ValueError("Neuron IDs can't be negative.")

        # Store data dimensions.
        self._n_trials = max_trial + 1
        self._n_neurons = max_neuron + 1

        # Store trial id, neuron id, and spike time for each spike.
        self._trials = trials
        self._spiketimes = spiketimes
        self._neurons = neurons

        # Sort spikes by trial id. The up front cost of this computation is
        # often worth it for faster shifting and indexing.
        self._sort_by_trial()

    @property
    def n_trials(self):
        return self._n_trials

    @property
    def n_neurons(self):
        return self._n_neurons

    @property
    def n_spikes(self):
        return len(self._n_trials)

    def shift_each_trial(self, shifts):
        """
        Adds a constant offset to spike times on each trial.

        Parameters
        ----------
        shifts : array-like
            1d array or sequence specifying the offset/shift to apply to all
            spike times on each trial.
        """
        if shifts != self._n_trials:
            raise ValueError('Input must match number of trials.')
        _shift_each_trial(self._trials, self._spiketimes, shifts)

    def reorder_trials(self, trial_map):
        """
        Re-indexes all spikes according to trial permutation.
        """
        self._reorder(self._trials, trial_map)

    def reorder_neurons(self, neuron_map):
        """
        Re-indexes all spikes according to trial permutation.
        """
        self._reorder(self._neurons, neuron_map)

    def _reorder(self, arr, arr_map):
        """
        Re-indexes spikes for generic array.
        """
        idx = np.empty(self.n_spikes, dtype=int)
        _get_reindexing(arr, arr_map, idx)
        self._reindex_spikes(idx)

    def filter_trials(self, kept_trials):
        """
        Filter out trials by integer id.
        """
        self._filter(self._trials, kept_trials)
        # rename trial ids so that they are zero-indexed, continuous integers.
        self._trials = rankdata(self._trials, method='dense') - 1

    def filter_neurons(self, kept_neurons):
        """
        Filter out neurons by integer id.
        """
        self._filter(self._neurons, kept_neurons)
        # rename neuron ids so that they are zero-indexed, continuous integers.
        self._neurons = rankdata(self._neurons, method='dense') - 1

    def _filter(self, arr, ids):
        """
        Filter out elements for generic array.
        """
        idx = np.zeros(self.n_spikes, dtype=bool)
        _get_filtered_indexing(self._arr, ids, idx)
        self._reindex_spikes(idx)

    def _reindex_spikes(self, idx):
        """
        Re-indexes all spikes according to input.
        """
        self._trials = self._trials[idx]
        self._spiketimes = self._spiketimes[idx]
        self._neurons = self._neurons[idx]
        self._sort_by_trial()

    def _sort_by_trial(self):
        """
        Permutes the order of data so that trial ids are sorted.
        """
        idx = np.argsort(self._trials)
        self._trials = np.ascontiguousarray(self._trials[idx])
        self._spiketimes = np.ascontiguousarray(self._spiketimes[idx])
        self._neurons = np.ascontiguousarray(self._neurons[idx])


@jit(nopython=True)
def _get_reindexing(arr, arr_map, idx):
    """
    Computes re-indexing array to remap values in 'arr'.
    """
    for i, x in enumerate(arr):
        idx[i] = arr_map[x]


@jit(nopython=True)
def _get_filtered_indexing(arr, kept_values, idx):
    """
    Computes re-indexing array to keep only subset of values in 'arr'.
    """
    for i, x in enumerate(arr):
        if x in kept_values:
            idx[i] = True


@jit(nopython=True)
def _shift_each_trial(trials, times, shifts):
    """
    Overwrites 'times' array by adding a trial-specific temporal shift to each
    spike time.
    """
    for k in trials:
        times[k] += shifts[k]
