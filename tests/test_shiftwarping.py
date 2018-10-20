"""
Tests for ShiftWarping models.
"""

import numpy as np
from affinewarp.shiftwarp import _fill_WtW


def test_WtW():

    # test WtW has correct number of counts
    K, T = 100, 101
    shifts = np.random.randint(-25, 26, size=K)
    WtW = np.zeros(T)
    _fill_WtW(shifts, WtW)
    assert int(WtW.sum()) == (K * T)
