"""
Functions to generate synthetic data for testing functionality.
"""

from .piecewisewarp import PiecewiseWarping
import numpy as np
from scipy.ndimage import gaussian_filter1d


def piecewise_warped_data(
        n_trials=120, n_timepoints=100, n_neurons=50, n_knots=1,
        knot_mutation_scale=0.1, clip_y_knots=True, template_scale=3.0,
        template_base=0.0, template_drop=0.5, template_smoothness=5.0,
        noise_type="poisson", noise_scale=0.1, seed=None):
    """Generates data from the PiecewiseWarping model.

    Parameters
    ----------
    n_trials : int
        Number of trials in synthetic data.
    n_timepoints : int
        Number of timepoints per trial.
    n_neurons : int
        Number of neurons/features in the time series.
    n_knots : int
        Number of knots in the warping function.
    knot_mutation_scale : float
        Scale of noise added to warping function knots.
    clip_y_knots : bool
        If True, clip y coordinates on warping functions between zero and one.
    template_scale : float
        Scale of exponentially distributed template values (before smoothing).
    template_drop : float
        Probability of zeroing template values (spike-and-slab distribution).
    template_base : float
        Min value of ground-truth model template.
    template_smoothness : float
        Width of Gaussian smoothing on model template.
    noise_type : str
        Either "poisson" or "gaussian", specifies type of noise applied to warped
        template on every trial.
    noise_scale : float
        If noise_type == "gaussian" this is the standard deviation of noise.
    seed : int
        Used to initialize RandomState instance.

    Returns
    -------
    data : ndarray (trials x timepoints x features)
        Collection of time series.
    model : PiecewiseWarping
        Ground truth model.
    """

    # Initialize random state
    rs = np.random.RandomState(seed)

    # Create ground-truth model.
    model = PiecewiseWarping(n_knots=n_knots)

    # Initialize warping knots.
    model.initialize_warps(n_trials)

    # Mutate warping knots.
    if n_knots < 0:
        # Shift-only warping.
        xy_noise = rs.rand(n_trials, 1)
        xy_noise *= 2 * knot_mutation_scale
        xy_noise -= knot_mutation_scale
        x = np.column_stack((np.zeros(n_trials), np.ones(n_trials)))
        y = x + xy_noise

    else:
        # Linear or piecewise linear warping.
        x_noise = rs.randn(n_trials, n_knots + 2) * knot_mutation_scale
        x = model.x_knots + x_noise
        x.sort(axis=1)
        x = x - x[:, (0,)]
        x = x / x[:, (-1,)]

        y_noise = rs.randn(n_trials, n_knots + 2) * knot_mutation_scale
        y = model.y_knots + y_noise
        y.sort(axis=1)

    model.x_knots, model.y_knots = x, y

    # If desired, clip y_knots.
    if clip_y_knots:
        model.y_knots[:, 0] = 0.
        model.y_knots[:, -1] = 1.

    # Initialize template.
    template_shape = n_timepoints, n_neurons
    template = template_base + \
        rs.exponential(template_scale, size=template_shape) * \
        rs.binomial(1, 1 - template_drop, size=template_shape)
    template = gaussian_filter1d(template, template_smoothness, axis=0)
    template[0] *= 0.0
    template[-1] *= 0.0
    model.template = template

    # Generate data
    data = model.predict()
    if noise_type == 'poisson':
        data = rs.poisson(data)
    elif noise_type == 'gaussian':
        data += rs.normal(loc=0.0, scale=noise_scale)

    return data, model


def jittered_data(t=None, feature=None, n_trial=61, jitter=1, gain=0,
                  noise=0.05, seed=None, sort=False):
    """Add temporal shifts / ofssets to a time series dataset.

    Parameters
    ----------
    t : array_like (optional)
        vector of within-trial timepoints
        (default: linearly spaced timepoints from -5 to 5)
    feature : function  (optional)
        produces the time series for a trial, given a numeric offset.
        (default: gaussian curve)
    n_trial : int
        number of trials (default: 61)
    jitter : float
        standard deviation of trial-to-trial shifts in timing.
        (default: 1)
    gain : float
        standard deviation of trial-to-trial changes in amplitude
        (default: 0)
    noise : float
        scale of additive gaussian noise
        (default: 0.05)
    sort : bool
        If True, sort the trials by the jitter (default: False)
    seed : int
        seed for the random number generator
        (default: None)

    Returns
    -------
    canonical_feature : array_like
        vector of firing rates on a trial with zero jitter
    aligned_data : array_like
        n_trial x n_time x 1 array of de-jittered noisy data
    jittered_data : array_like
        n_trial x n_time x 1 array of firing rates with jitter and noise
    """

    # default time base
    if t is None:
        t = np.linspace(-5, 5, 150)

    # default feature
    if feature is None:
        def feature(tau): return np.exp(-(t-tau)**2)

    # noise matrix
    np.random.seed(seed)
    noise = noise*np.random.randn(n_trial, len(t))

    # generate jittered data
    gains = 1.0 + gain*np.random.randn(n_trial)
    shifts = jitter*np.random.randn(n_trial)
    if sort:
        shifts.sort()
    jittered_data = np.array([g*feature(s) for g, s in zip(gains, shifts)]) + noise

    # generate aligned data
    aligned_data = np.array([g*feature(0) for g in gains]) + noise

    return feature(0), np.atleast_3d(aligned_data), np.atleast_3d(jittered_data)
