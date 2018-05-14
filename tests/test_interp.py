import numpy as np
from affinewarp.interp import warp_penalties
from numpy.testing import assert_allclose


def test_warp_penalties():

    # identity warp should produce zero penalty
    for n in range(2, 10):
        X = np.linspace(0, 1, 10)
        Y = np.linspace(0, 1, 10)

        actual = warp_penalties(X[None, :], Y[None, :], np.empty(1))
        expected = np.array([0.0])

        assert_allclose(actual, expected, atol=1e-8)

    # warp shifted by s should produce penalty s
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




if __name__ == '__main__':
    test_warp_penalties()
