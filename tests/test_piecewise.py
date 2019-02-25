"""
Tests to check that piecewise warps are transforming data as expected.
"""

import pytest
import numpy as np
from affinewarp._optimizers import warp_penalties
from affinewarp import PiecewiseWarping, SpikeData
from numpy.testing import assert_allclose, assert_array_equal


# def test_monotonic_knots():
#     """
#     Test that warping functions remain monotonically increasing during random
#     search.
#     """
#     model = PiecewiseWarping()
#     data = np.random.randn(100, 101, 102)
#     model.fit(data, iterations=1, verbose=False)

#     for temperature in np.logspace(-3, 3, 100):
#         x, y = model._mutate_knots(temperature)
#         assert np.all(np.diff(x, axis=1) >= 0)
#         assert np.all(np.diff(y, axis=1) >= 0)


def test_warp_penalties():
    """
    Test that regularization on the warping functions (penalizing distance
    from identity) is correctly computed for some simple cases.
    """

    # Identity warp should produce zero penalty
    for n in range(2, 10):
        X = np.linspace(0, 1, 10)
        Y = np.linspace(0, 1, 10)

        actual = warp_penalties(X[None, :], Y[None, :], np.empty(1))
        expected = np.array([0.0])

        assert_allclose(actual, expected, atol=1e-8)

    # Warp shifted by s should produce penalty s
    for s in (-.5, .5):
        for n in range(2, 10):
            X = np.linspace(0, 1, n)
            Y = np.linspace(0, 1, n) + s

            actual = warp_penalties(X[None, :], Y[None, :], np.empty(1))
            expected = np.array([abs(s)])

            assert_allclose(actual, expected, atol=1e-8)

    # warp scaled by 2
    for n in range(2, 10):
        X = np.linspace(0, 1, n)
        Y = np.linspace(0, 1, n)*2

        actual = warp_penalties(X[None, :], Y[None, :], np.empty(1))
        expected = np.array([0.5])

        assert_allclose(actual, expected, atol=1e-8)

    # horizontal warp
    for n in range(2, 10):
        X = np.linspace(0, 1, n)
        Y = np.ones(n) * 0.5

        actual = warp_penalties(X[None, :], Y[None, :], np.empty(1))
        expected = np.array([0.25])

        assert_allclose(actual, expected, atol=1e-8)


def test_identity_transform():
    """
    Test that identity warping functions do not change spike times.
    """

    # dense data
    model = PiecewiseWarping()
    binned = np.random.randn(10, 11, 12)
    model.fit(binned, iterations=0, verbose=False)
    assert_array_equal(binned, model.transform(binned))

    # sparse data
    k, t, n = np.where(np.random.randn(10, 11, 12) > 1.5)
    spikedata = SpikeData(k, t, n, tmin=0, tmax=12)
    warped = model.transform(spikedata)
    assert_array_equal(spikedata.trials, warped.trials)
    assert_allclose(spikedata.spiketimes, warped.spiketimes)
    assert_array_equal(spikedata.neurons, warped.neurons)


@pytest.mark.parametrize('s', [.5, 1.0, 2.0])
def test_scaling(s):
    """
    Test that linear warps with different slopes appropriately scale spike
    times.
    """

    # Create random dataset.
    K, T, N = 10, 11, 12
    k, t, n = np.where(np.random.randn(K, T, N) > 1.5)
    data = SpikeData(k, t, n, tmin=0, tmax=T)

    # Create linear warping model.
    model = PiecewiseWarping()
    model.x_knots = np.column_stack((np.zeros(K), np.ones(K)))
    model.y_knots = np.column_stack((np.zeros(K), np.full(K, 1 / s)))

    # Test spike dataset is appropriately transformed.
    wdata = model.transform(data)
    assert_allclose(wdata.spiketimes, data.spiketimes / s)

    # Test event_transform(...) has the same functionality.
    wst = model.event_transform(data.trials, data.fractional_spiketimes)
    assert_allclose(wst, data.fractional_spiketimes / s)
