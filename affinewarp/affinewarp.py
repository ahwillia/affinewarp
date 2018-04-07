import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy as sci
from tqdm import trange, tqdm
from .utils import modf, _reduce_sum_assign, _reduce_sum_assign_matrix
from .tridiag import trisolve


class AffineWarping(object):
    """Represents a collection of time series, each with an affine time warp.
    """
    def __init__(self, data, max_shift=.2, max_scale=1.5, boundary=0):
        """
        Params
        ------
        data (ndarray) : n_trials x n_timepoints x n_features
        """

        # data dimensions
        self.data = data
        self.n_trials = data.shape[0]
        self.n_timepoints = data.shape[1]
        self.n_features = data.shape[2]

        # model options
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.boundary = boundary

        # trial-average under affine warping (initialize to random trial)
        self.template = data[np.random.randint(0, self.n_trials)].copy()
        self.tref = np.linspace(0, 1, self.n_timepoints)
        self.dt = self.tref[1]-self.tref[0]
        self.apply_warp = interp1d(self.tref, self.template, axis=0, assume_sorted=True)

        # initialize shifts (taus) and scales (betas) of warping functions
        self.taus = self._sample_taus()
        self.betas = self._sample_betas()
        self.warping_funcs = self._compute_warping_funcs()

        # reconstructed data
        self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        self.resids = self.reconstruction - self.data

        # initial loss for each trial calculation
        self.losses = sci.linalg.norm(self.resids, axis=(1, 2))
        self.loss_hist = [np.mean(self.losses)]

    def _compute_warping_funcs(self, taus=None, betas=None):
        if taus is None:
            taus = self.taus
        if betas is None:
            betas = self.betas
        return np.clip((self.tref * betas[:, None]) - taus[:, None], 0, 1)

    def _sample_taus(self):
        """Randomly sample shifts
        """
        lo, hi = -self.max_shift, self.max_shift
        return np.random.uniform(lo, hi, size=self.n_trials)

    def _sample_betas(self):
        """Randomly sample slopes
        """
        lo, hi = np.log(1/self.max_scale), np.log(self.max_scale)
        return np.exp(np.random.uniform(lo, hi, size=self.n_trials))

    def fit_warps(self, iterations=100):

        # preallocate room for sampled reconstructions
        recon = np.empty_like(self.reconstruction)

        for i in trange(iterations):
            # randomly sample warping functions
            taus = self._sample_taus()
            betas = self._sample_betas()
            warps = self._compute_warping_funcs(taus, betas)

            # warp data and compute new losses
            for k, t in enumerate(warps):
                recon[k] = self.apply_warp(t)
            losses = sci.linalg.norm(recon - self.data, axis=(1, 2))

            # update warping parameters for trials with improved loss
            idx = losses < self.losses
            self.losses[idx] = losses[idx]
            self.taus[idx] = taus[idx]
            self.betas[idx] = betas[idx]
            self.warping_funcs[idx] = warps[idx]
            self.reconstruction[idx] = recon[idx]
            self.loss_hist.append(np.mean(self.losses))

    def fit_template(self):
        # compute normal equations
        T = self.n_timepoints
        WtW_d0 = np.full(T, 1e-5)
        WtW_d1 = np.zeros(T-1)
        WtX = np.zeros((T, self.n_features))
        for wfunc, Xk in zip(self.warping_funcs, self.data):
            lam, i = modf(wfunc * (T-1))

            _reduce_sum_assign(WtW_d0, i, (1-lam)**2)
            _reduce_sum_assign(WtW_d0, i+1, lam**2)
            _reduce_sum_assign(WtW_d1, i, lam*(1-lam))

            _reduce_sum_assign_matrix(WtX, i, (1-lam[:, None]) * Xk)
            _reduce_sum_assign_matrix(WtX, i+1, lam[:, None] * Xk)

        # update template
        self.template = trisolve(WtW_d1, WtW_d0, WtW_d1, WtX)

        if self.boundary is not None:
            self.template[0, :] = self.boundary
            self.template[-1, :] = self.boundary

        self.apply_warp = interp1d(self.tref, self.template, axis=0, assume_sorted=True)

        # update loss
        self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        self.resids = self.reconstruction - self.data
        self.losses = sci.linalg.norm(self.resids, axis=(1, 2))
        self.loss_hist.append(np.mean(self.losses))

        return self.template

    def transform(self, fill_value=0):
        warped_data = np.empty_like(self.data)
        for k in range(self.n_trials):
            f = interp1d(self.warping_funcs[k], self.tref, kind='slinear',
                         axis=0, bounds_error=False, fill_value='extrapolate',
                         assume_sorted=True)
            g = interp1d(self.tref, self.data[k], axis=0, bounds_error=False,
                         fill_value=fill_value, assume_sorted=True)
            warped_data[k] = g(f(self.tref))
        return warped_data
