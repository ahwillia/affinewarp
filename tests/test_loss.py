"""
Tests for loss calculations.
"""

import pytest
import numpy as np
from affinewarp.datasets import jittered_data
from affinewarp import ShiftWarping, PiecewiseWarping


@pytest.mark.parametrize('model', [
    ShiftWarping(loss='quadratic'),
    PiecewiseWarping(n_knots=0, loss='quadratic'),
    PiecewiseWarping(n_knots=1, loss='quadratic'),
])
@pytest.mark.parametrize('data', [
    jittered_data()[-1],
])
def test_quad_loss_init(model, data):

    # Fit model template with identity warping functions.
    model.fit(data, iterations=0)

    # Compute expected loss.
    resids = model.template[None, :, :] - data
    loss_per_trial = np.mean(resids ** 2, axis=(1, 2))
    expected_loss = np.mean(loss_per_trial)

    # Compute error
    absolute_error = np.abs(expected_loss - model.loss_hist[0])
    assert absolute_error < 1e-6
