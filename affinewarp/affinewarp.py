import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy as sci
from tqdm import trange, tqdm
from .utils import modf, _reduce_sum_assign, _reduce_sum_assign_matrix
from tslearn.barycenters import SoftDTWBarycenter
from .tridiag import trisolve
from .interp import bcast_interp, interp_knots
import time


class AffineWarping(object):
    """Piecewise Affine Time Warping applied to an analog (dense) time series.
    """
    def __init__(self, data, q1=.3, q2=.15, boundary=0, n_knots=0,
                 l2_smoothness=0):
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

        self.l2_smoothness = l2_smoothness

        # trial-average under affine warping (initialize to random trial)
        # self.template = SoftDTWBarycenter(gamma=1, max_iter=100).fit(data)
        self.template = data[np.random.randint(0, self.n_trials)].astype(float)
        # self.template = data.mean(axis=0)
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
        self._new_losses = np.empty_like(self.losses)

    def _sample_knots(self, n):
        """Randomly sample warping functions.
        """
        x = np.column_stack((np.zeros(n),
                             np.sort(np.random.rand(n, self.n_knots)),
                             np.ones(n)))
        y = np.column_stack((np.zeros(n),
                             np.sort(np.random.rand(n, self.n_knots)),
                             np.ones(n)))
        y = self.q1*y + (1-self.q1)*x

        y0 = np.random.uniform(-self.q2, self.q2, size=(n, 1))
        y1 = np.random.uniform(1-self.q2, 1+self.q2, size=(n, 1))
        y = (y1-y0)*y + y0

        return x, y

    def fit(self, iterations=10, warp_iterations=20):

        pbar = trange(iterations)
        for it in pbar:
            l0 = self.loss_hist[-1]
            self.fit_warps(warp_iterations)
            self.fit_template()
            imp = (l0-self.loss_hist[-1])/l0
            pbar.set_description('Loss improvement: {0:.2f}%'.format(imp*100))

        return self

    def fit_warps(self, iterations=20):

        for i in range(iterations):
            # randomly sample warping functions
            X, Y = self._sample_knots(self.n_trials)

            bcast_interp(self.tref, X, Y, self._new_warps,
                         self.template, self._new_losses, self.losses,
                         self.data)

            # update warping parameters for trials with improved loss
            idx = self._new_losses < self.losses
            self.losses[idx] = self._new_losses[idx]
            self.x_knots[idx] = X[idx]
            self.y_knots[idx] = Y[idx]
            self.warping_funcs[idx] = self._new_warps[idx]


    def fit_template(self):
        # compute normal equations
        T = self.n_timepoints

        if self.l2_smoothness > 0:
            # coefficent matrix for the template update reduce to a
            # banded matrix with 5 diagonals.
            WtW = np.zeros((3, T))
            WtW[0, 2:] = 1.0 * self.l2_smoothness
            WtW[1, 2:] = -4.0 * self.l2_smoothness
            WtW[1, 1] = -2.0 * self.l2_smoothness
            WtW[2, 2:] = 6.0 * self.l2_smoothness
            WtW[2, 1] = 5.0 * self.l2_smoothness
            WtW[2, 0] = 1.0 * self.l2_smoothness
            _WtW = WtW[1:, :]
        else:
            # coefficent matrix for the template update reduce to a
            # banded matrix with 3 diagonals.
            WtW = np.zeros((2, T))
            _WtW = WtW

        # WtW_d0 = np.full(T, 1e-3)
        # WtW_d1 = np.zeros(T-1)
        WtX = np.zeros((T, self.n_features))

        for wfunc, Xk in zip(self.warping_funcs, self.data):
            lam, i = modf(wfunc * (T-1))

            _reduce_sum_assign(_WtW[1, :], i, (1-lam)**2)
            _reduce_sum_assign(_WtW[1, :], i+1, lam**2)
            _reduce_sum_assign(_WtW[0, 1:], i, lam*(1-lam))

            _reduce_sum_assign_matrix(WtX, i, (1-lam[:, None]) * Xk)
            _reduce_sum_assign_matrix(WtX, i+1, lam[:, None] * Xk)

        self.template = sci.linalg.solveh_banded(WtW, WtX, overwrite_ab=True, overwrite_b=True)

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

    # def transform_events(self, k, t):
    #     """Apply inverse warping functions to events.
    #     """
    #     assert len(k) == len(t)

    #     T = self.n_timepoints
    #     lam, i = modf(t * (T-1))

    #     t_out = self.warping_funcs[k, i]*(1-lam) + \
    #         self.warping_funcs[k, (i+1) % T]*(lam)

    #     return t_out

    def transform_events(self, k, t):
        """Apply inverse warping functions to events.
        """
        assert len(k) == len(t)

        return interp_knots(self.x_knots, self.y_knots, k, t)
