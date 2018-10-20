"""
Tests for Poisson loss.
"""
import pytest
from affinewarp.datasets import piecewise_warped_data
from affinewarp._optimizers import PoissonObjective
from scipy.optimize import approx_fprime
import numpy as np

TOL = 1e-5
N_TRIALS = 3
N_NEURONS = 2
DATA, MODEL = piecewise_warped_data(n_trials=N_TRIALS, n_neurons=N_NEURONS)

X_KNOTS = [
    np.tile(np.linspace(0, 1, MODEL.n_knots + 2), (N_TRIALS, 1)),
    MODEL.x_knots,
]
Y_KNOTS = [
    np.tile(np.linspace(0, 1, MODEL.n_knots + 2), (N_TRIALS, 1)),
    MODEL.y_knots,
]
X0 = [
    DATA.mean(axis=0).ravel(),
    np.median(DATA, axis=0).ravel(),
]


@pytest.mark.parametrize('x0', X0)
@pytest.mark.parametrize('x_knots', X_KNOTS)
@pytest.mark.parametrize('y_knots', Y_KNOTS)
@pytest.mark.parametrize('l2_smoothness_scale', [0.0, 1.0, 10.0])
@pytest.mark.parametrize('l2_scale', [0.0, 1.0, 10.0])
def test_gradients(x0, x_knots, y_knots, l2_smoothness_scale, l2_scale):

    obj = PoissonObjective(DATA, l2_smoothness_scale, l2_scale,
                           x_knots=x_knots, y_knots=y_knots)

    def f(x):
        return obj(x)[0]

    numerical_grad = approx_fprime(x0, f, 1e-5)
    grad = obj(x0)[1]

    norm_error = np.linalg.norm(numerical_grad - grad)
    norm_scale = np.linalg.norm(numerical_grad + grad)
    assert (norm_error / norm_scale) < TOL
