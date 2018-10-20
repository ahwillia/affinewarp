"""
Tests for Poisson loss.
"""
import pytest
from affinewarp.datasets import piecewise_warped_data
from affinewarp._optimizers import PoissonObjective
from scipy.optimize import check_grad, approx_fprime
import numpy as np

TOL = 1e-5


@pytest.mark.parametrize('l2_smoothness_scale', [0.0, 1.0, 10.0])
@pytest.mark.parametrize('l2_scale', [0.0, 1.0, 10.0])
def test_gradients(l2_smoothness_scale, l2_scale):

    data, model = piecewise_warped_data(n_trials=3, n_neurons=2)

    obj = PoissonObjective(data, l2_smoothness_scale, l2_scale,
                           x_knots=model.x_knots, y_knots=model.y_knots)

    def f(x):
        return obj(x)[0]

    x0 = data.mean(axis=0).ravel()
    numerical_grad = approx_fprime(x0, f, 1e-5)

    norm_error = np.linalg.norm(numerical_grad - obj(x0)[1])
    norm_scale = np.linalg.norm(numerical_grad + obj(x0)[1])
    assert (norm_error / norm_scale) < TOL
