import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy as sci
from tqdm import trange, tqdm
from .utils import modf, _reduce_sum_assign, _reduce_sum_assign_matrix
from .tridiag import trisolve
import pdb


class AffineWarping(object):
    """Represents a collection of time series, each with an affine time warp.
    """
    def __init__(self, data, q1=.3, q2=0, boundary=0, n_knots=0):
        """
        Params
        ------
        data (ndarray) : n_trials x n_timepoints x n_features
        """

        # check inputs
        # if n_knots < 0:
            # raise ValueError('Number of knots must be nonnegative.')

        # data dimensions
        self.data = data
        self.n_trials = data.shape[0]
        self.n_timepoints = data.shape[1]
        self.n_features = data.shape[2]

        # model options
        self.boundary = boundary
        self.n_knots = n_knots
        self.q1 = q1
        self.q2 = q2

        # trial-average under affine warping (initialize to random trial)
        # self.template = data[np.random.randint(0, self.n_trials)].copy()
        self.template = data.mean(axis=0)
        self.tref = np.linspace(0, 1, self.n_timepoints)
        self.dt = self.tref[1]-self.tref[0]
        self.apply_warp = interp1d(self.tref, self.template, axis=0, assume_sorted=True)

        # initialize warping functions
        self.x_knots, self.y_knots = self._sample_knots(self.n_trials)
        self.warping_funcs = self._compute_warping_funcs(self.x_knots, self.y_knots)

        # reconstructed data
        self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        self.resids = self.reconstruction - self.data

        # initial loss for each trial calculation
        self.losses = sci.linalg.norm(self.resids, axis=(1, 2))
        self.loss_hist = [np.mean(self.losses)]

    def _compute_warping_funcs(self, X, Y):
        return np.clip([np.interp(self.tref, _x, _y) for _x, _y in zip(X, Y)], 0, 1)
        # Alternate solution not involving a for loop:
        #   dX = np.diff(X, axis=1)
        #   dY = np.diff(Y, axis=1)
        #   w = np.clip((self.tref - X[:, :-1, None])/dX[:, :, None], 0, 1)
        #   return np.clip(Y[:, [0]] + np.sum(w*dY[:, :, None], axis=1), 0, 1)

    def _sample_knots(self, n):
        """Randomly sample warping functions
        """
        if self.n_knots < 0:
            x = np.column_stack((np.zeros(n), np.ones(n)))
            x += np.random.uniform(-self.q2, self.q2, size=(n, 1))
            return x, x

        x = np.column_stack((np.zeros(n), np.sort(np.random.rand(n, self.n_knots)), np.ones(n)))
        y = np.column_stack((np.zeros(n), np.sort(np.random.rand(n, self.n_knots)), np.ones(n)))
        y = self.q1*y + (1-self.q1)*x

        y0 = np.random.uniform(-self.q2, self.q2, size=(n, 1))
        y1 = np.random.uniform(1-self.q2, 1+self.q2, size=(n, 1))
        y = (y1-y0)*y + y0

        return x, y

    def fit(self, iterations=10):

        for it in trange(iterations):
            self.fit_warps()
            self.fit_template()

    def fit_warps(self, iterations=100, desc=None):

        # preallocate room for sampled reconstructions
        recon = np.empty_like(self.reconstruction)

        for i in range(iterations):
            # randomly sample warping functions
            X, Y = self._sample_knots(self.n_trials)
            warps = self._compute_warping_funcs(X, Y)

            # warp data and compute new losses
            for k, t in enumerate(warps):
                recon[k] = self.apply_warp(t)
            losses = sci.linalg.norm(recon - self.data, axis=(1, 2))

            # update warping parameters for trials with improved loss
            idx = losses < self.losses
            self.losses[idx] = losses[idx]
            self.x_knots[idx] = X[idx]
            self.y_knots[idx] = Y[idx]
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

    def transform(self, data=None):
        data = self.data if data is None else data

        if len(data) != self.n_trials:
            raise ValueError('Input must have the same number of trials as fitted data.')

        warped_data = np.empty_like(data)
        for k in range(self.n_trials):
            f = interp1d(self.warping_funcs[k], self.tref, kind='slinear',
                         axis=0, bounds_error=False, fill_value='extrapolate',
                         assume_sorted=True)
            g = interp1d(self.tref, data[k], axis=0, bounds_error=False,
                         fill_value=self.boundary, assume_sorted=True)
            warped_data[k] = g(f(self.tref))
        return warped_data
