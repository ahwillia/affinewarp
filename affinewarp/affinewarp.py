import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy as sci
from tqdm import trange, tqdm
from .utils import bin_count_data, modf, _reduce_sum_assign, _reduce_sum_assign_matrix
from tslearn.barycenters import SoftDTWBarycenter
from .interp import bcast_interp, interp_knots

INIT_ERROR = RuntimeError(
    "Model is not initialized. Call 'model.initialize_fit(...)' before "
    "calling 'model.fit(...)' or 'model.predict(...)'"
)

DATA_ERROR = ValueError(
    "Input must be 3d array holding (trials x timepoints x features). Data "
    "for each trial may be encoded as a sparse matrix; in this case the "
    "dimensions of the sparse matrices must match and the data is binned "
    "based on the 'nbins' parameter passed to 'model.fit(...)'."
)


class AffineWarping(object):
    """Piecewise Affine Time Warping applied to an analog (dense) time series.
    """
    def __init__(self, q1=.3, q2=.15, boundary=0, n_knots=0,
                 l2_smoothness=0):

        # check inputs
        if n_knots < 0:
            raise ValueError('Number of knots must be nonnegative.')

        # model options
        self.boundary = boundary
        self.n_knots = n_knots
        self.q1 = q1
        self.q2 = q2
        self.l2_smoothness = l2_smoothness

        # initialized in self.initialize_fit(...)
        self.data = None
        self.template = None
        self._losses = None
        self.n_trials = None
        self.n_timepoints = None
        self.n_features = None
        self.loss_hist = None

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

    def initialize_fit(self, data, nbins=200):
        """Preprocesses data by binning spike times
        """

        # check data dimensions as input
        data = np.asarray(data)

        # check if dense array
        if data.ndim == 3 and np.issubdtype(data.dtype, np.number):
            self.data = data

        # interpret input data as sparse count data
        else:
            # attempt to bin count data
            try:
                data = bin_count_data(data, nbins)
                self.data = data
            except ValueError:
                raise DATA_ERROR

        # data dimensions
        self.n_trials = data.shape[0]
        self.n_timepoints = data.shape[1]
        self.n_features = data.shape[2]

        # initialize template
        self.template = data[np.random.randint(self.n_trials)].astype(float)

        # time base
        self.tref = np.linspace(0, 1, self.n_timepoints)
        self.dt = self.tref[1] - self.tref[0]

        # trial-average under affine warping (initialize to random trial)
        self.apply_warp = interp1d(self.tref, self.template, axis=0,
                                   assume_sorted=True)

        # initialize warping functions to identity
        self.x_knots = np.tile(
            np.linspace(0, 1, self.n_knots+2),
            (self.n_trials, 1)
        )
        self.y_knots = self.x_knots.copy()  # TODO - remove copy?
        self.warping_funcs = np.tile(self.tref, (self.n_trials, 1))

        # update loss
        self._losses = sci.linalg.norm(self.predict() - data, axis=(1, 2))
        self.loss_hist = [np.mean(self._losses)]

        # arrays used in fit_warps function
        self._new_warps = np.empty_like(self.warping_funcs)
        self._new_losses = np.empty_like(self._losses)

    def fit(self, iterations=10, warp_iterations=20, verbose=True):
        """Fits warping functions and model template

        Parameters
        ----------
        spiketimes (container of array-like)
            list of spike times within each trial
        neurons (container of array-like)
            list of neuron index for each spike
        """

        if self.data is None:
            raise INIT_ERROR

        # progress bar
        pbar = trange(iterations) if verbose else range(iterations)

        # fit model
        for it in pbar:
            last_loss = self.loss_hist[-1]
            self._fit_warps(warp_iterations)
            self._fit_template()

            # display progress
            if verbose:
                imp = 100 * (last_loss - self.loss_hist[-1]) / last_loss
                pbar.set_description('Loss improvement: {0:.2f}%'.format(imp))

        return self

    def dump_params(self):
        """Returns a list of model parameters for storage

        Note: see 'AffineWarping.load_params(...)' companion function.
        """
        return (self.template, self.x_knots, self.y_knots, self.loss_hist)

    def load_params(self, data, params):
        """Loads parameters to AffineWarping.

        Note: see 'AffineWarping.dump_params(...)' companion function.
        """

        # loaded model parameters
        template, x_knots, y_knots, loss_hist = params

        if data.shape[0] != len(x_knots) or data.shape[0] != len(y_knots):
            raise ValueError('number of trials in data and loaded parameters '
                             ' do not match')

        if data[0].shape[1] != template.shape[1]:
            raise ValueError('number of measured features in data and loaded '
                             'parameters do not match')

        # initialize fitting method with data
        self.initialize_fit(data, nbins=template.shape[0])

        # load main parameters
        self.template = template
        self.x_knots = x_knots
        self.y_knots = y_knots
        self.loss_hist = loss_hist

        # reconstitute warping funcs
        self.warping_funcs =\
            [np.interp(self.tref, xk, yk) for xk, yk in zip(x_knots, y_knots)]

        # update warping procedure so that it uses the new template
        self.apply_warp = interp1d(
            self.tref, self.template, axis=0, assume_sorted=True
        )

        # compute losses for each trial
        self._losses = sci.linalg.norm(self.predict() - self.data, axis=(1, 2))

    def _fit_warps(self, iterations=20):
        """Fit warping functions by random search.
        """

        for i in range(iterations):
            # randomly sample warping functions
            X, Y = self._sample_knots(self.n_trials)

            bcast_interp(self.tref, X, Y, self._new_warps,
                         self.template, self._new_losses, self._losses,
                         self.data)

            # update warping parameters for trials with improved loss
            idx = self._new_losses < self._losses
            self._losses[idx] = self._new_losses[idx]
            self.x_knots[idx] = X[idx]
            self.y_knots[idx] = Y[idx]
            self.warping_funcs[idx] = self._new_warps[idx]

    def _fit_template(self):
        """Fit template by least squares.
        """

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

        WtX = np.zeros((T, self.n_features))

        for wfunc, Xk in zip(self.warping_funcs, self.data):
            lam, i = modf(wfunc * (T-1))

            _reduce_sum_assign(_WtW[1, :], i, (1-lam)**2)
            _reduce_sum_assign(_WtW[1, :], i+1, lam**2)
            _reduce_sum_assign(_WtW[0, 1:], i, lam*(1-lam))

            _reduce_sum_assign_matrix(WtX, i, (1-lam[:, None]) * Xk)
            _reduce_sum_assign_matrix(WtX, i+1, lam[:, None] * Xk)

        # solve WtW * template = WtX
        self.template = sci.linalg.solveh_banded(
            WtW, WtX, overwrite_ab=True, overwrite_b=True
        )

        # enforce boundary conditions
        if self.boundary is not None:
            self.template[0, :] = self.boundary
            self.template[-1, :] = self.boundary

        # update warping procedure so that it uses the new template
        self.apply_warp = interp1d(
            self.tref, self.template, axis=0, assume_sorted=True
        )

        # update reconstruction and evaluate loss
        self._losses = sci.linalg.norm(self.predict() - self.data, axis=(1, 2))
        self.loss_hist.append(np.mean(self._losses))

        return self.template

    def predict(self):
        # check initialization
        if self.warping_funcs is None:
            raise INIT_ERROR

        # apply warping functions to template
        return np.asarray([self.apply_warp(t) for t in self.warping_funcs])

    # def transform_events(self, k, t):
    #     """Apply inverse warping functions to events.
    #     """
    #     assert len(k) == len(t)

    #     T = self.nbins
    #     lam, i = modf(t * (T-1))

    #     t_out = self.warping_funcs[k, i]*(1-lam) + \
    #         self.warping_funcs[k, (i+1) % T]*(lam)

    #     return t_out

    def transform_sparse_matrix(self, data):
        new_data = []
        T = data[0].shape[1]
        for k, trial in enumerate(data):

            t, n = trial.nonzero()
            new_t = self.transform_events(np.full(len(t), k), t).astype(int)

            t == j

            for j in range(trial.shape[1]):
                idx = (n == j)
                _t, counts = np.unique(new_t[idx], return_counts=True)

            np.unique(new_t)

    def transform_events(self, k, t):
        """Apply inverse warping functions to events.
        """
        assert len(k) == len(t)

        return interp_knots(self.x_knots, self.y_knots, k, t)
