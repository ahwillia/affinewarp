"""
Specialized code for handling spike data.
"""
import itertools
import numpy as np
from numba import jit
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
        tmin: float
            Beginning of each trial (same time units as 'spiketimes').
        tmax: float
            End of each trial (same time units as 'spiketimes').
        """

        # Treat inputs as numpy arrays.
        self.trials = np.asarray(trials)
        self.spiketimes = np.asarray(spiketimes).astype(float)
        self.neurons = np.asarray(neurons)
        self.tmin = tmin
        self.tmax = tmax

        # All inputs must be 1-dimensional.
        if not (self.trials.ndim == self.spiketimes.ndim == self.neurons.ndim == 1):
            raise ValueError("All inputs must be 1D arrays.")

        # Trial and neuron ids must be nonnegative integers.
        if not np.issubdtype(self.trials.dtype, np.integer):
            raise ValueError("Trial IDs must be integers.")
        if not np.issubdtype(self.neurons.dtype, np.integer):
            raise ValueError("Neuron IDs must be integers.")

        min_trial, max_trial = min_max_1d(self.trials)
        if min_trial < 0:
            raise ValueError("Trial IDs can't be negative.")
        min_neuron, max_neuron = min_max_1d(self.neurons)
        if min_neuron < 0:
            raise ValueError("Neuron IDs can't be negative.")

        # Store data dimensions.
        self.n_trials = max_trial + 1
        self.n_neurons = max_neuron + 1

        # Sort spikes by trial id. The up front cost of this computation is
        # often worth it for faster shifting and indexing.
        # self._sort_by_trial()

        # Stores spike times normalized between zero and one (fraction of time
        # within trial).
        self._frac_spiketimes = None

        # Essential data need to make a copy
        self._data = (self.trials, self.spiketimes, self.neurons, tmin, tmax)

    @property
    def fractional_spiketimes(self):
        if self._frac_spiketimes is None:
            self._frac_spiketimes = (self.spiketimes - self.tmin) / (self.tmax - self.tmin)
        return self._frac_spiketimes

    @property
    def n_spikes(self):
        return len(self.trials)

    @property
    def shape(self):
        return (self.n_trials, self.tmax - self.tmin, self.n_neurons)

    def bin_spikes(self, n_bins):
        """
        Bins spikes into dense array.

        Parameters
        ----------
        n_bins : int
            Number of timebins per trial.

        Returns
        -------
        binned : ndarray
            Binned spike counts (n_trials x n_bins x n_neurons).
        """

        # allocate space for result
        shape = (self.n_trials, n_bins, self.n_neurons)
        binned = np.zeros(shape, dtype=int)

        # compute bin for each spike
        bin_ids = (self.fractional_spiketimes * (n_bins-1)).astype(int)

        # add up all spike counts and return result
        _fast_bin(binned, self.trials, bin_ids, self.neurons)
        return binned

    def shift_each_trial_by_fraction(self, fractional_shifts, copy=True):
        """
        Adds an offset to spike times on each trial.

        Shifts are given in terms of fraction of trial duration. For example,
        a shift of 0.1 corresponds to a 10 percent shift later in the trial,
        while a shift of -0.1 corresponds to a 10 percent shift earlier in
        the trial.

        Parameters
        ----------
        fractional_shifts : array-like
            1d array specifying shifts.
        """
        # Convert fractional shifts to real time units before shifting.
        shifts = fractional_shifts * (self.tmax - self.tmin)
        return self.shift_each_trial_by_constant(shifts, copy=copy)

    def shift_each_trial_by_constant(self, absolute_shifts, copy=True):
        """
        Adds an offset to spike times on each trial.

        Shifts are given in absolute unites (same as self.tmin and self.tmax).

        Parameters
        ----------
        absolute_shifts : array-like
            1d array specifying shifts.
        """

        # Check inputs.
        if len(absolute_shifts) != self.n_trials:
            raise ValueError('Input must match number of trials.')

        if copy:
            result = self.copy()
        else:
            # Set fractional spike times to None, as these need to be
            # recalculated after shifting.
            self._frac_spiketimes = None
            result = self

        # Apply shifts.
        _shift_each_trial(result.trials, result.spiketimes, absolute_shifts)
        return result

    def reorder_trials(self, trial_indices, copy=True):
        """
        Re-indexes all spikes according to trial permutation. Indexing
        semantics are the same as Numpy standard.
        """
        if any(np.unique(trial_indices) != np.arange(self.n_trials)):
            raise ValueError('Indices must be a permutation of trials. See '
                             'SpikeData.filter_trials to select subsets of '
                             'trials.')
        result = self.copy() if copy else self
        # argsort indices to get position/destination for reindexing
        _reindex(result.trials, np.argsort(trial_indices))
        return result

    def reorder_neurons(self, neuron_indices, copy=True):
        """
        Re-indexes all spikes according to neuron permutation. Indexing
        semantics are the same as Numpy standard.
        """
        if any(np.unique(neuron_indices) != np.arange(self.n_neurons)):
            raise ValueError('Indices must be a permutation of neurons. See '
                             'SpikeData.filter_neurons to select subsets of '
                             'neurons.')
        result = self.copy() if copy else self
        # argsort indices to get position/destination for reindexing
        _reindex(result.neurons, np.argsort(neuron_indices))
        return result

    def filter_trials(self, kept_trials, copy=True):
        """
        Filter out trials by integer id.
        """
        result = self.copy() if copy else self
        result._filter(result.trials, kept_trials)
        # rename trial ids so that they are zero-indexed, continuous integers.
        result.trials = rankdata(result.trials, method='dense') - 1
        return result

    def filter_neurons(self, kept_neurons, copy=True):
        """
        Filter out neurons by integer id.
        """
        result = self.copy() if copy else self
        result._filter(result.neurons, kept_neurons)
        # rename neuron ids so that they are zero-indexed, continuous integers.
        result.neurons = rankdata(result.neurons, method='dense') - 1
        return result

    def _filter(self, arr, ids):
        """
        Filter out elements for generic array.
        """
        idx = np.zeros(self.n_spikes, dtype=bool)
        _get_filtered_indexing(self._arr, ids, idx)
        self.trials = self.trials[idx]
        self.spiketimes = self.spiketimes[idx]
        self.neurons = self.neurons[idx]
        self._sort_by_trial()

    def _sort_by_trial(self):
        """
        Permutes the order of data so that trial ids are sorted.
        """
        idx = np.argsort(self.trials)
        self.trials = np.ascontiguousarray(self.trials[idx])
        self.spiketimes = np.ascontiguousarray(self.spiketimes[idx])
        self.neurons = np.ascontiguousarray(self.neurons[idx])

    def copy(self):
        return type(self)(self.trials.copy(), self.spiketimes.copy(),
                          self.neurons.copy(), self.tmin, self.tmax)


@jit(nopython=True)
def _fast_bin(counts, trials, bins, neurons):
    """
    Given coordinates of spikes, compile binned spike counts.
    """
    for i, j, k in zip(trials, bins, neurons):
        counts[i, j, k] += 1


@jit(nopython=True)
def _reindex(arr, arr_map):
    """
    Computes re-indexing array to remap values in 'arr'.
    """
    for i, x in enumerate(arr):
        arr[i] = arr_map[x]


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
    for spike_index, k in enumerate(trials):
        times[spike_index] -= shifts[k]


@jit(nopython=True)
def min_max_1d(arr):
    """
    Return maximum and minimum value of a 1d array.
    """
    vmax = arr[0]
    vmin = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > vmax:
            vmax = arr[i]
        elif arr[i] < vmin:
            vmin = arr[i]
    return vmin, vmax
