from numba import jit


class SpikeData(object):
    """docstring for SpikeData"""
    def __init__(self, trials, times, neurons, metadata=dict()):

        # treat inputs as 1d numpy arrays
        trials = np.asarray(trials).ravel().astype(int)
        times = np.asarray(times).ravel()
        neurons = np.asarray(neurons).ravel().astype(int)

        # check dimensions
        if not (len(trials) == len(times) == len(neurons)):
            raise ValueError('Data vectors are of unequal length.')

        # sort spikes by trial number
        if not is_sorted(trials):
            i = trials.argsort()
            trials = trials[i]
            times = times[i]
            neurons = trials[i]

        # number trial sequentially, starting at zero.
        _sequentially_renumber(trials)

        # renumber neurons sequentially (like trials), but since neurons isn't
        # sorted we have to resort to scipy.stats.rankdata.
        neurons = rankdata(neurons, method='dense') - 1
        self.n_neurons = np.max(neurons)

        # store spike time data
        self.trials = trials
        self.neurons = neurons
        self.times = times
        self.metadata = metadata

    def remove_trials(self, trial_indices):
        """Remove specified trials from dataset
        """

        # convert from boolean masking array to
        if trial_indices.dtype == bool:
            trial_indices = np.argwhere(trial_indices).ravel()

        # make sure trial ids are sorted and pass some sanity checks
        idx = np.unique(trial_indices)

        assert len(idx) == len(trial_indices)
        assert idx[-1] < len(trial_indices)
        assert idx[0] >= 0

        # new metadata
        metadata = {k: v[idx] for k, v in self.metadata.items()}

        # allocate space for new spike times
        trials = self.trials.copy()
        times = self.times.copy()
        neurons = self.neurons.copy()

        # filter out trials
        n = _filter_spikes(trials, times, neurons, idx)

        # create new spike dataset instance
        return SpikeData(trials[:n], times[:n], neurons[:n], metadata)

    def remove_neurons(self, neuron_indices):
        """Remove specified neurons from dataset
        """

        # convert from boolean masking array to
        if neuron_indices.dtype == bool:
            neuron_indices = np.argwhere(neuron_indices).ravel()

        # make sure trial ids are sorted and pass some sanity checks
        idx = np.unique(neuron_indices)

        assert len(idx) == len(neuron_indices)
        assert idx[-1] < len(neuron_indices)
        assert idx[0] >= 0

        # allocate space for new spike times
        trials = np.empty_like(self.trials)
        times = np.empty_like(self.times)
        neurons = np.empty_like(self.neurons)

        # sort data by neurons for fast filter operation
        srt = np.argsort(self.neurons)
        trials[:] = self.trials[srt]
        times[:] = self.times[srt]
        neurons[:] = self.neurons[srt]

        # filter out neurons
        n = _filter_spikes(neurons, times, trials, idx)

        # create new spike dataset instance.
        #     - metadata is left unchanged, as it only contains trial data
        return SpikeData(trials[:n], times[:n], neurons[:n], metadata)

    def clip_spiketimes(self, tmin=None, tmax=None):

        if tmax is not None and tmin is not None:
            idx = (self.times <= tmax) & (self.times >= tmin)
        elif tmax is not None:
            idx = self.times <= tmax
        elif tmin is not None:
            idx = self.times >= tmin
        else:
            raise ValueError('Must specify either tmin or tmax, or both.')

        return SpikeData(self.trials[idx], self.times[idx],
                         self.neurons[idx], self.metadata)

    def bin_spikes(self, nbins, tmin=None, tmax=None):

        # determine trial start and stop
        tmin = times.min()-1e-6 if tmin is None else tmin
        tmax = times.max()+1e-6 if tmax is None else tmax

        # compute bin size
        binsize = (tmax - tmin) / nbins

        # bin spikes
        binned = np.zeros((self.n_trials, nbins, self.n_neurons))
        _fast_bin(binned, trials, times, neurons, tmin, binsize)

        # return (trials x timebins x neurons) array of binned spike counts
        return binned

    @property
    def n_trials(self):
        # number of trials is always the last element in trial vector
        return self.trials[-1]


@jit(nopython=True)
def _filter_spikes(trials, times, neurons, idx):
    """Throw away spike

    Args
    ----
    trials : 1d array, holds trial number of each spike (dtype: int)
    times : 1d array, holds within-trial time of each spike (dtype: float)
    neurons : 1d array, holds neuron associated with each spike (dtype: int)
    idx : pre-sorted 1d array, trial numbers retained in dataset (dtype: int)

    Returns
    -------
    k : number of spikes
    """

    i = 0  # counts over spikes in dataset
    j = 0  # counts over trial labels
    k = 0  # counts over retained spikes

    while i < len(trials):

        while (j < len(idx)) and (trials[i] > idx[j]):
            j += 1

        if trials[i] == idx[j]:

            if k != i:
                trials[k] = trials[i]
                times[k] = times[i]
                neurons[k] = neurons[i]

            k += 1

        i += 1

    return k


@jit(nopython=True)
def _sequentially_renumber(X):
    """Given a sorted array of ints, renumber elements as sequential integers.

    Note: X is updated inplace
    """
    n = 0
    for i in range(X.size-1):
        if X[i+1] < X[i]:
            X[i] = n
            n = n+1
        else:
            X[i] = n
    X[-1] = n


@jit(nopython=True)
def is_sorted(X):
    """Checks if numpy array is sorted
    """
    for i in range(X.size-1):
        if X[i+1] < X[i]:
            return False
    return True


@jit(nopython=True)
def _fast_bin(out, k, t, n, tmin, binsize):
    for i in range(len(k)):
        # compute bin
        b = int((t[i] - tmin) / binsize)

        # ignore any spike outside of tmin and tmax
        if (b < out.shape[1]) and (b > 0):
            out[k[i], b, n[i]] += 1
