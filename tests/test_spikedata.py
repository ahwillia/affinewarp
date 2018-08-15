import numpy as np
from affinewarp.spikedata import SpikeData
from numpy.testing import assert_array_equal


def test_bin_spikes():
    trials = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    spiketimes = [0, 1, 1, 2, 3, 3, 4, 4, 1, 2, 2, 3, 3, 3, 4, 4]
    neurons = [0, 0, 1, 0, 0, 2, 1, 2, 0, 0, 1, 0, 1, 2, 0, 2]

    # Create dataset with single spikes in each bin and compare to np.where
    data = SpikeData(trials, spiketimes, neurons, tmin=0, tmax=4)
    binned = data.bin_spikes(n_bins=5)
    _trials, _times, _neurons = np.where(binned)
    assert binned.sum() == len(trials)
    assert_array_equal(trials, _trials)
    assert_array_equal(spiketimes, _times)
    assert_array_equal(neurons, _neurons)
    assert_array_equal(np.unique(binned), [0, 1])

    # Test same dataset with two spikes in each bin.
    data2 = SpikeData(
        trials+trials, spiketimes+spiketimes, neurons+neurons, tmin=0, tmax=4)
    binned2 = data2.bin_spikes(n_bins=5)
    _trials, _times, _neurons = np.where(binned2)
    assert binned2.sum() == 2*len(trials)
    assert_array_equal(trials, _trials)
    assert_array_equal(spiketimes, _times)
    assert_array_equal(neurons, _neurons)
    assert_array_equal(np.unique(binned2), [0, 2])

    # Test that spikes outside of [tmin, tmax] are ignored during binning.
    n_spikes = 1000
    trials = np.random.randint(10, size=n_spikes)
    neurons = np.random.randint(11, size=n_spikes)
    spiketimes = np.random.rand(n_spikes)
    d1 = SpikeData(trials, spiketimes, neurons, tmin=0.25, tmax=0.75)
    idx = (spiketimes >= d1.tmin) & (spiketimes <= d1.tmax)
    d2 = SpikeData(trials[idx], spiketimes[idx],
                   neurons[idx], tmin=d1.tmin, tmax=d1.tmax)
    for n_bins in [10, 50, 100]:
        assert_array_equal(d1.bin_spikes(n_bins), d2.bin_spikes(n_bins))


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
