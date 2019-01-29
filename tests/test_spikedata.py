"""
Tests for SpikeData class.
"""

import pytest
import numpy as np
from affinewarp.spikedata import SpikeData
from numpy.testing import assert_array_equal


def test_bin_spikes():

    # Simpled dataset with one spike per bin.
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


def test_reorder_trials():

    # Generate random dataset (trials x time x neuron)
    X = (np.random.rand(30, 40, 50) > .5).astype(int)
    X[0, 0, 0] = 1
    X[-1, -1, -1] = 1

    # Add no spikes on some trials.
    X[10:15] = 0

    # Construct SpikeData instance.
    trials, spiketimes, neurons = np.where(X)
    data = SpikeData(trials, spiketimes, neurons, tmin=0, tmax=X.shape[1]-1)

    # Bin spikes at resolution of X.
    binned = data.bin_spikes(X.shape[1])
    assert_array_equal(X, binned)

    # Permute X and SpikeData by the same reordering. Test for equality.
    kk = np.random.permutation(X.shape[0])
    Xprm = X[kk]
    permuted = data.reorder_trials(kk)
    assert_array_equal(Xprm, permuted.bin_spikes(X.shape[1]))

    assert permuted.n_trials == data.n_trials


def test_select_neurons():

    # Generate random dataset (trials x time x neuron)
    nk, nt, nn = 30, 40, 50
    X = (np.random.rand(nk, nt, nn) > .5).astype(int)
    X[0, 0, 0] = 1
    X[-1, -1, -1] = 1

    # Zero-out at least one neuron.
    X[:, :, 1] *= 0

    # Construct SpikeData instance.
    trials, spiketimes, neurons = np.where(X)
    data = SpikeData(trials, spiketimes, neurons, tmin=0, tmax=nt-1)

    # Bin spikes at resolution of X.
    assert_array_equal(X, data.bin_spikes(nt))

    # Subselect neurons and test for equality with sub-selecting X.
    for n in range(X.shape[-1]):
        smalldata = data.select_neurons(n)
        binned = X[:, :, (n,)]
        if smalldata.n_neurons == 0:
            assert_array_equal(binned, np.zeros_like(binned))
        else:
            assert_array_equal(binned, smalldata.bin_spikes(nt))
