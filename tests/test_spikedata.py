import numpy as np
from affinewarp.spikedata import SpikeData
from numpy.testing import assert_array_equal


def test_bin_spikes():
    trials = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    spiketimes = [0, 1, 1, 2, 3, 3, 4, 4, 1, 2, 2, 3, 3, 3, 4, 4]
    neurons = [0, 0, 1, 0, 0, 2, 1, 2, 0, 0, 1, 0, 1, 2, 0, 2]

    data = SpikeData(trials, spiketimes, neurons, tmin=0, tmax=4)
    binned = data.bin_spikes(n_bins=5)
    _trials, _times, _neurons = np.where(binned)
    assert binned.sum() == len(trials)
    assert_array_equal(trials, _trials)
    assert_array_equal(spiketimes, _times)
    assert_array_equal(neurons, _neurons)
    assert_array_equal(np.unique(binned), [0, 1])

    # create dataset with two spikes in each bin
    data2 = SpikeData(
        trials+trials, spiketimes+spiketimes, neurons+neurons, tmin=0, tmax=4)
    binned2 = data2.bin_spikes(n_bins=5)
    _trials, _times, _neurons = np.where(binned2)
    assert binned2.sum() == 2*len(trials)
    assert_array_equal(trials, _trials)
    assert_array_equal(spiketimes, _times)
    assert_array_equal(neurons, _neurons)
    assert_array_equal(np.unique(binned2), [0, 2])


def test_reordering():
    X = (np.random.rand(30, 40, 50) > .5).astype(int)
    X[0, 0, 0] = 1
    X[-1, -1, -1] = 1
    trials, spiketimes, neurons = np.where(X)
    data = SpikeData(trials, spiketimes, neurons, tmin=0, tmax=X.shape[1]-1)

    binned = data.bin_spikes(X.shape[1])
    assert_array_equal(X, binned)

    kk = np.random.permutation(X.shape[0])
    Xprm = X[kk]
    permuted = data.reorder_trials(kk)
    assert_array_equal(Xprm, permuted.bin_spikes(X.shape[1]))


if __name__ == '__main__':
    test_bin_spikes()
    test_reordering()
