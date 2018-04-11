import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy as sci
from tqdm import trange, tqdm
from .utils import modf, _reduce_sum_assign, _reduce_sum_assign_matrix
from .tridiag import trisolve
from .interp import bcast_interp
import time


class AffineWarping(object):
    """Represents a collection of time series, each with an affine time warp.
    """
    def __init__(self, data, q1=.3, q2=.15, boundary=0, n_knots=0):
        """
        Params
        ------
        data (ndarray) : n_trials x n_timepoints x n_features
        """

        # check inputs
        if n_knots < 0:
            raise ValueError('Number of knots must be nonnegative.')

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

        # initialize warping functions to identity
        self.x_knots = np.tile(np.linspace(0, 1, n_knots+2), (self.n_trials, 1))
        self.y_knots = self.x_knots.copy()
        self.warping_funcs = np.tile(self.tref, (self.n_trials, 1))

        # reconstructed data
        self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        self.resids = self.reconstruction - self.data

        # initial loss for each trial calculation
        self.losses = sci.linalg.norm(self.resids, axis=(1, 2))
        self.loss_hist = [np.mean(self.losses)]

        # used during fit update
        self._new_warps = np.empty_like(self.warping_funcs)
        self._new_pred = np.empty_like(self.reconstruction)
        self._new_losses = np.empty_like(self.losses)

    def _sample_knots(self, n):
        """Randomly sample warping functions
        """
        x = np.column_stack((np.zeros(n), np.sort(np.random.rand(n, self.n_knots)), np.ones(n)))
        y = np.column_stack((np.zeros(n), np.sort(np.random.rand(n, self.n_knots)), np.ones(n)))
        y = self.q1*y + (1-self.q1)*x

        y0 = np.random.uniform(-self.q2, self.q2, size=(n, 1))
        y1 = np.random.uniform(1-self.q2, 1+self.q2, size=(n, 1))
        y = (y1-y0)*y + y0

        self.warp_time = 0
        self.fit_time = 0

        return x, y

    def fit(self, iterations=10, warp_iterations=20):

        pbar = trange(iterations)
        for it in pbar:
            l0 = self.loss_hist[-1]
            self.fit_warps(warp_iterations)
            self.fit_template()
            imp = (l0-self.loss_hist[-1])/l0
            pbar.set_description('Loss improvement: {}%'.format(imp*100))

    def fit_warps(self, iterations=20):

        for i in range(iterations):
            # randomly sample warping functions
            X, Y = self._sample_knots(self.n_trials)

            bcast_interp(self.tref, X, Y, self._new_warps, self._new_pred,
                         self.template, self._new_losses, self.losses,
                         self.data)

            # update warping parameters for trials with improved loss
            idx = self._new_losses < self.losses
            self.losses[idx] = self._new_losses[idx]
            self.x_knots[idx] = X[idx]
            self.y_knots[idx] = Y[idx]
            self.warping_funcs[idx] = self._new_warps[idx]
            # self.reconstruction[idx] = self._new_pred[idx]
            # self.loss_hist.append(np.mean(self.losses))

            self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
            self.resids = self.reconstruction - self.data
            self.losses = sci.linalg.norm(self.resids, axis=(1, 2))
            self.loss_hist.append(np.mean(self.losses))

    def fit_template(self):
        # compute normal equations
        T = self.n_timepoints
        WtW_d0 = np.full(T, 1e-20)
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

    def transform(self, data=None, trials=None):

        # by default, warp the training data
        data = self.data if data is None else data
        trials = range(self.n_trials) if trials is None else trials

        if not isinstance(data, np.ndarray):
            raise ValueError("Argument 'data' should be an ndarray")

        # check for singleton dimension
        elif data.ndim == 2 and data.shape[1] == 1:
            data = data.ravel()

        if data.ndim == 1:
            # interpret data as events
            t = data / self.n_timepoints
            warped_data = np.empty((len(trials), self.n_timepoints))
            for i, k in enumerate(trials):
                warped_data[i] = np.interp(t[k], self.tref, self.warping_funcs[k])

        else:
            # interpret data as a dense tensor
            _tref = np.linspace(0, 1, data.shape[1])

            # apply inverse warping function
            warped_data = np.empty((len(trials), self.n_timepoints))
            for i, k in enumerate(trials):
                f = interp1d(self.warping_funcs[k], _tref, kind='slinear',
                             axis=0, bounds_error=False,
                             fill_value='extrapolate', assume_sorted=True)
                g = interp1d(_tref, data[k], axis=0, bounds_error=False,
                             fill_value=self.boundary, assume_sorted=True)
                warped_data[i] = g(f(_tref))

        return warped_data

    def sort_by_warping(self, ts):
        return np.argsort(self.warping_funcs[:, ts])
