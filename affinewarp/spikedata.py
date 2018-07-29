"""
Specialized code for handling spike data.
"""
import itertools
import numpy as np
import numba
from scipy.stats import rankdata


class SpikeData(object):
    """
    Represents a collection multi-dimensional spike train data.

    Attributes
    ----------
    trials : ndarray of ints
        Integer ids/indices for trial of each spike.
    spiketimes : ndarray of floats
        Float timepoints for each spike (within-trial time).
    neurons : ndarray of ints
        Integer ids/indices for neuron/electrode firing each spike.
    tmin : float
        Beginning of each trial (same time units as 'spiketimes').
    tmax : float
        End of each trial (same time units as 'spiketimes').
    n_trials : int
        Number of trials in the dataset.
    n_neurons : int
        Number of neurons in the dataset.
    n_spikes : int
        Total number of spikes across all neurons and trials.
    shape : tuple (int, float, int)
        Number of trials, duration of each trial, number of neurons.
    fractional_spiketimes : ndarray of floats
        Same as spiketimes, but expressed as a fraction of time within trial.
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
        self.sort_spikes()

        # Stores spike times normalized between zero and one (fraction of time
        # within trial). Initialized to None and computed on demand.
        self._frac_spiketimes = None

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

    def spikes_per_trial(self):
        """
        Computes number of spikes fired by all neurons on each trial.
        """
        vals, counts = np.unique(self.trials, return_counts=True)
        trial_counts = np.zeros(self.n_trials, dtype=int)
        trial_counts[vals] = counts
        return trial_counts

    def spikes_per_neuron(self):
        """
        Computes number of spikes fired by each neuron across all trials.
        """
        vals, counts = np.unique(self.neurons, return_counts=True)
        neuron_counts = np.zeros(self.n_neurons, dtype=int)
        neuron_counts[vals] = counts
        return neuron_counts

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

        Raises
        ------
        ValueError: If n_bins is not a positive integer.
        """
        if n_bins <= 0 or not np.issubdtype(type(n_bins), np.integer):
            raise ValueError("Expected 'n_bins' to be a positive integer, but "
                             "saw {}".format(n_bins))

        # Compute bin for each spike.
        bin_ids = (self.fractional_spiketimes * (n_bins-1) + .5).astype(int)

        # Allocate space for result. If any spiketimes are outside of the
        # interval [tmin, tmax] then increase number of time bins linearly.
        shape = (self.n_trials, max(n_bins, bin_ids.max()+1), self.n_neurons)
        binned = np.zeros(shape, dtype=int)

        # Add up all spike counts and return result.
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
        _trialcopy = result.trials.copy()
        _reindex(result.trials, np.argsort(trial_indices))
        result.sort_spikes()
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
        result.sort_spikes()
        return result

    def select_trials(self, kept_trials, copy=True):
        """
        Filter out trials by integer id.
        """
        if not np.iterable(kept_trials):
            kept_trials = (kept_trials,)
        kept_trials = np.asarray(kept_trials)
        if kept_trials.dtype == bool:
            kept_trials = np.where(kept_trials)[0]
        elif not is_sorted(kept_trials):
            raise ValueError("kept_trials must be sorted.")
        result = self.copy() if copy else self
        result._filter(result.trials, kept_trials)
        result.sort_spikes()
        result.n_trials = result.trials[-1] + 1
        return result

    def select_neurons(self, kept_neurons, copy=True):
        """
        Filter out neurons by integer id.
        """
        if not np.iterable(kept_neurons):
            kept_neurons = (kept_neurons,)
        kept_neurons = np.asarray(kept_neurons)
        if kept_neurons.dtype == bool:
            kept_neurons = np.where(kept_neurons)[0]
        elif not is_sorted(kept_neurons):
            raise ValueError("kept_neurons must be sorted.")
        result = self.copy() if copy else self
        result._filter(result.neurons, kept_neurons)
        result.sort_spikes()
        result.n_neurons = result.neurons.max() + 1
        return result

    def add_trial(self, new_times, new_neurons):
        """
        Adds a new trial of spikes to the dataset.

        Parameters
        ----------
        new_times: array-like
            1D sequence holding spike times.
        new_neurons:
            1D sequence holding the id of the neuron firing each spike.

        Raises
        ------
        ValueError: if 'new_times' and 'new_neurons' do not have the same
            length or are not 1-dimensional.
        """
        # Check inputs.
        new_times = np.asarray(new_times).ravel()
        new_neurons = np.asarray(new_neurons).ravel()
        if len(new_times) != len(new_neurons):
            raise ValueError('Mismatched array lengths.')

        # Sort new spikes in lexographic order.
        idx = np.argsort(new_times)
        new_times = new_times[idx]
        new_neurons = new_neurons[idx]

        # Concatenate spike coordinates.
        new_trials = np.full(len(times), self.n_trials)
        self.trials = np.concatenate((self.trials, new_trials))
        self.spiketimes = np.concatenate((self.spiketimes, new_times))
        self.neurons = np.concatenate((self.neurons, new_neurons))

        # Increment number of trials in the dataset.
        self.n_trials += 1

        # If new a neuron is added, update number of neurons in the dataset.
        max_neuron = new_neurons.max()
        if max_neuron >= self.n_neurons:
            self.n_neurons = max_neuron + 1

    def _filter(self, arr, kept_values):
        """
        Filter out elements for generic array.
        """
        idx = np.zeros(self.n_spikes, dtype=bool)
        _get_filtered_indexing(arr, kept_values, idx)
        self.trials = self.trials[idx]
        self.spiketimes = self.spiketimes[idx]
        self.neurons = self.neurons[idx]

    def sort_spikes(self):
        """
        Permutes the order of data so that trial ids are sorted.
        """
        idx = np.lexsort((self.neurons, self.spiketimes, self.trials))
        self.trials = np.ascontiguousarray(self.trials[idx])
        self.spiketimes = np.ascontiguousarray(self.spiketimes[idx])
        self.neurons = np.ascontiguousarray(self.neurons[idx])

    def copy(self):
        return type(self)(self.trials.copy(), self.spiketimes.copy(),
                          self.neurons.copy(), self.tmin, self.tmax)


@numba.jit(nopython=True)
def _fast_bin(counts, trials, bins, neurons):
    """
    Given coordinates of spikes, compile binned spike counts.
    """
    for i, j, k in zip(trials, bins, neurons):
        counts[i, j, k] += 1


@numba.jit(nopython=True)
def _reindex(arr, arr_map):
    """
    Computes re-indexing array to remap values in 'arr'.
    """
    for i, x in enumerate(arr):
        arr[i] = arr_map[x]


@numba.jit(nopython=True)
def _get_filtered_indexing(arr, kept_values, idx):
    """
    Computes re-indexing array to keep only subset of values in 'arr'.
    """
    for i, x in enumerate(arr):
        arr[i] = binary_search(kept_values, x)
        if arr[i] >= 0:
            idx[i] = True


@numba.jit(nopython=True)
def _shift_each_trial(trials, times, shifts):
    """
    Overwrites 'times' array by adding a trial-specific temporal shift to each
    spike time.
    """
    for spike_index, k in enumerate(trials):
        times[spike_index] -= shifts[k]


@numba.jit(nopython=True)
def is_sorted(arr):
    """
    Returns True if arr is a sorted numpy array and False otherwise.
    """
    for i in range(arr.size-1):
        if arr[i+1] < arr[i]:
            return False
    return True


@numba.jit(nopython=True)
def binary_search(arr, item):
    """
    Returns position of 'item' in 1d array 'arr', assumes that 'arr' is sorted.
    If 'arr' does not contain 'item', returns -1.
    """
    i = 0
    j = arr.size-1
    while i <= j:
        mid = (i + j) // 2
        if arr[mid] == item:
            return mid
        elif item < arr[mid]:
            j = mid - 1
        else:
            i = mid + 1
    return -1


@numba.jit(nopython=True)
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
