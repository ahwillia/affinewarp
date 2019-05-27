"""
Tests that optimization is being performed correctly.
"""

import pytest
import numpy as np
from affinewarp import PiecewiseWarping
from scipy.ndimage import gaussian_filter1d
from numpy.testing import assert_allclose


def test_identity_warps_gauss_errors():
    """
    Test that model fits converge to trial-average with identity warps.
    """

    # Create small, synthetic dataset
    n_trials = 10
    n_timepoints = 11
    n_units = 3
    data = np.random.randn(n_trials, n_timepoints, n_units)

    # Define model
    model = PiecewiseWarping(
        n_knots=0,
        warp_reg_scale=0.0,
        smoothness_reg_scale=0.0,
        l2_reg_scale=0.0,
    )

    # Fit model template without updating warps.
    model.initialize_warps(n_trials)
    model._fit_template(data, np.arange(n_trials))

    assert_allclose(model.template, data.mean(axis=0), rtol=1e-4)
