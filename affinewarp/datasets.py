"""Simple datasets for testing functionality."""

import numpy as np


# TODO(poolio)
# ------------
# Add function that generates spikes from piecewise linear time warped process.
# Could also just generate binned spike counts.


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
