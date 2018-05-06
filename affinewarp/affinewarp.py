import numpy as np
from scipy.interpolate import interp1d
import scipy as sci
from tqdm import trange, tqdm
from .utils import modf, _fast_template_grams, quad_loss
from .interp import warp_with_loss, densewarp, sparsewarp
from numba import jit
import sparse


class AffineWarping(object):
    """Piecewise Affine Time Warping applied to an analog (dense) time series.
    """
    def __init__(self, q1=.3, q2=.15, boundary=0, n_knots=0, l2_smoothness=0):

        # check inputs
        if n_knots < 0:
            raise ValueError('Number of knots must be nonnegative.')

        # model options
        self.n_knots = n_knots
        self.q1 = q1
        self.q2 = q2
        self.l2_smoothness = l2_smoothness

        # initialize model now if data is provided
        self.template = None
        self.warping_funcs = None

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

    def fit(self, data, **kwargs):
        """Initializes warping functions and model template and begin fitting.
        """

        # check data dimensions as input
        data = np.asarray(data)

        # check if dense array
        if data.ndim != 3 or not np.issubdtype(data.dtype, np.number):
            raise ValueError("'data' must be provided as a numpy ndarray "
                             "(neurons x timepoints x trials) holding binned "
                             "spike data.")

        # data dimensions
        K = data.shape[0]
        T = data.shape[1]
        N = data.shape[2]

        # initialize template
        self.template = data[np.random.randint(K)].astype(float)

        # time base
        self.tref = np.linspace(0, 1, T)

        # initialize warping functions to identity
        self.x_knots = np.tile(
            np.linspace(0, 1, self.n_knots+2),
            (K, 1)
        )
        self.y_knots = self.x_knots.copy()  # TODO - remove copy?
        self.warping_funcs = np.tile(self.tref, (K, 1))

        # update loss
        self._losses = quad_loss(self.predict(), data)
        self.loss_hist = [np.mean(self._losses)]

        # arrays used in fit_warps function
        self._new_warps = np.empty_like(self.warping_funcs)
        self._new_losses = np.empty_like(self._losses)

        # call fitting function
        self.continue_fit(data, **kwargs)

    def continue_fit(self, data, iterations=10, warp_iterations=20,
                     verbose=True):
        """Continues optimization of warps and template (no initialization).
        """

        # check that model is initialized.
        if self.template is None:
            raise ValueError("Model not initialized. Need to call "
                             "'AffineWarping.fit(...)' before calling "
                             "'AffineWarping.continue_fit(...)'.")

        if data.shape[-1] != self.template.shape[1]:
            raise ValueError('Dimension mismatch.')

        # progress bar
        pbar = trange(iterations) if verbose else range(iterations)

        # fit model
        for it in pbar:
            last_loss = self.loss_hist[-1]

            # Note: roughly equal computation time for 20 warp iterations.
            self.fit_warps(data, warp_iterations)
            self.fit_template(data)

            # display progress
            if verbose:
                imp = 100 * (last_loss - self.loss_hist[-1]) / last_loss
                pbar.set_description('Loss improvement: {0:.2f}%'.format(imp))

        return self

    def dump_params(self):
        """Returns a list of model parameters for storage
        """
        return {
            'template': self.template,
            'x_knots': self.x_knots,
            'y_knots': self.y_knots,
            'loss_hist': self.loss_hist,
            'l2_smoothness': self.l2_smoothness,
            'q1': self.q1,
            'q2': self.q2
        }

    def fit_warps(self, data, iterations=20, neurons=None):
        """Fit warping functions by random search.
        """

        if neurons is None:
            neurons = np.arange(data.shape[2])

        for i in range(iterations):
            # randomly sample warping functions
            X, Y = self._sample_knots(data.shape[0])

            # Note: this is the bulk of computation time.
            warp_with_loss(self.tref, X, Y, self._new_warps,
                           self.template, self._new_losses, self._losses,
                           data, neurons, _elemwise_quad)

            # update warping parameters for trials with improved loss
            idx = self._new_losses < self._losses
            self._losses[idx] = self._new_losses[idx]
            self.x_knots[idx] = X[idx]
            self.y_knots[idx] = Y[idx]
            self.warping_funcs[idx] = self._new_warps[idx]

    def fit_template(self, data, trials=None):
        """Fit template by least squares.
        """

        if trials is None:
            trials = slice(None)

        # compute normal equations
        T = data.shape[1]
        N = data.shape[2]

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
            _WtW = WtW[1:, :]  # makes _reduce_sum_assign target the right row.
        else:
            # coefficent matrix for the template update reduce to a
            # banded matrix with 3 diagonals.
            WtW = np.zeros((2, T))
            _WtW = WtW

        WtX = np.zeros((T, data.shape[-1]))

        lam, i = modf(self.warping_funcs[trials] * (T-1))
        lam, i = lam.ravel(), i.ravel()

        _fast_template_grams(_WtW, WtX, data.reshape(-1, N), lam, i)

        # solve WtW * template = WtX
        self.template = sci.linalg.solveh_banded(
            WtW, WtX, overwrite_ab=True, overwrite_b=True
        )

        # update reconstruction and evaluate loss
        self._losses = quad_loss(self.predict(), data)
        self.loss_hist.append(np.mean(self._losses))

        return self.template

    def predict(self):
        # check initialization
        if self.warping_funcs is None:
            raise ValueError("Model not initialized. Need to call "
                             "'AffineWarping.fit(...)' before calling "
                             "'AffineWarping.predict(...)'.")

        # apply warping functions to template
        f = interp1d(self.tref, self.template, axis=0, assume_sorted=True)
        return np.asarray([f(t) for t in self.warping_funcs])

    def transform(self, X):
        """Apply inverse warping functions to spike data
        """

        # check initialization
        if self.warping_funcs is None:
            raise ValueError("Model not initialized. Need to call "
                             "'AffineWarping.fit(...)' before calling "
                             "'AffineWarping.transform(...)'.")

        # add append new axis to 2d array if necessary
        if X.ndim == 2:
            X = X[:, :, None]
        elif X.ndim != 3:
            raise ValueError('Input should be 2d or 3d array.')

        # check that first axis of X matches n_trials
        if X.shape[0] != len(self.warping_funcs):
            raise ValueError('Number of trials in the input does not match '
                             'the number of trials in the fitted model.')

        # length of time axis undergoing warping
        T = X.shape[1]

        # sparse array transform
        if isinstance(X, sparse.SparseArray):

            # indices of sparse entries
            trials, times, neurons = sparse.where(X)

            # find warped time
            w = sparsewarp(self.x_knots, self.y_knots, trials, times / T)

            # return data as a new COO array
            wtimes = (w * T).astype(int)

            # throw away out of bounds spikes
            # TODO: add option to expand the dimensions instead
            i = (wtimes < T) & (wtimes >= 0)

            return sparse.COO([trials[i], wtimes[i], neurons[i]],
                              data=X.data[i], shape=X.shape)

        # dense array transform
        else:
            X = np.asarray(X)
            return densewarp(self.y_knots, self.x_knots, X, np.empty_like(X))


# LOSS FUNCTIONS #

# elementwise quadratic loss
@jit(nopython=True)
def _elemwise_quad(x, y):
    r = x - y
    return np.dot(r.ravel(), r.ravel())
