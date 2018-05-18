import numpy as np
import scipy as sci
from tqdm import trange, tqdm
from .utils import _diff_gramian
from sklearn.utils.validation import check_is_fitted
from .spikedata import get_spike_coords, get_spike_shape, is_spikedata
from numba import jit
import sparse


class AffineWarping(object):
    """Piecewise Affine Time Warping applied to an analog (dense) time series.
    """
    def __init__(self, n_knots=0, warpreg=0, l2_smoothness=0,
                 min_temp=-2, max_temp=-1):

        # check inputs
        if n_knots < 0:
            raise ValueError('Number of knots must be nonnegative.')

        # model options
        self.n_knots = n_knots
        self.warpreg = warpreg
        self.l2_smoothness = l2_smoothness
        self.min_temp = min_temp
        self.max_temp = max_temp

        # initialize model now if data is provided
        self.template = None
        self.x_knots = None
        self.y_knots = None

    def _mutate_knots(self, temperature):
        """
        Produces a new candidate set of warping functions (as parameterized by
        x and y coordinates of the knots) by randomly perturbing the current
        model parameters.

        Parameters
        ----------
        temperature : scalar
            Standard deviation of perturbation to the knots.
        """
        x, y = self.x_knots.copy(), self.y_knots.copy()
        K, P = x.shape
        y += np.random.randn(K, P) * temperature
        if self.n_knots > 0:
            x[:, 1:-1] += np.random.randn(K, self.n_knots) * temperature
        return _force_monotonic_knots(x, y)

    def fit(self, data, **kwargs):
        """
        Initializes warping functions and model template and begin fitting.
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
        self.template = data.mean(axis=0).astype(float)

        # time base
        self.tref = np.linspace(0, 1, T)

        # initialize warping functions to identity
        self.x_knots = np.tile(
            np.linspace(0, 1, self.n_knots+2),
            (K, 1)
        )
        self.y_knots = self.x_knots.copy()

        # compute initial loss
        self._losses = np.zeros(K)
        warp_with_quadloss(self.x_knots, self.y_knots, self.template,
                           self._losses, self._losses,
                           data, early_stop=False)

        # initial warp penalties and loss storage
        self._penalties = np.zeros(K)
        self.loss_hist = [np.mean(self._losses)]

        # arrays used in fit_warps function
        self._new_losses = np.empty_like(self._losses)
        self._new_penalties = np.empty_like(self._losses)

        # call fitting function
        self.continue_fit(data, **kwargs)

    def continue_fit(self, data, iterations=10, warp_iterations=20,
                     fit_template=True, verbose=True):
        """
        Continues optimization of warps and template (no initialization).

        Parameters
        ----------
        data : ndarray
            3d array (trials x times x features) holding signals to be fit.
        """
        check_is_fitted(self, 'template')

        # check inputs
        if data.shape[-1] != self.template.shape[1]:
            raise ValueError('Dimension mismatch.')

        # progress bar
        pbar = trange(iterations) if verbose else range(iterations)

        # fit model
        for it in pbar:

            # update warping functions
            self._fit_warps(data, warp_iterations)

            # user has option to only fit warps
            if fit_template:

                # update template
                self._fit_template(data)

                # update reconstruction and evaluate loss
                self._losses.fill(0.0)
                warp_with_quadloss(self.x_knots, self.y_knots, self.template,
                                   self._losses, self._losses,
                                   data, early_stop=False)

                # add warping penalty to losses
                if self.warpreg > 0:
                    self._losses += self._penalties

            # store objective function over time
            self.loss_hist.append(self._losses.mean())

            # display progress
            if verbose:
                imp = 100 * (self.loss_hist[-2] - self.loss_hist[-1]) / self.loss_hist[-2]
                pbar.set_description('Loss improvement: {0:.2f}%'.format(imp))

        return self

    def _fit_warps(self, data, iterations=20):
        """
        Fit warping functions by local random search. Typically, users should
        call either AffineWarping.fit(...) or AffineWarping.continue_fit(...)
        instead of this function.
        """

        # decay temperature within each epoch
        temperatures = np.logspace(self.min_temp, self.max_temp, iterations)

        # fit warps
        for temp in reversed(temperatures):

            # randomly sample warping functions
            X, Y = self._mutate_knots(temp)

            # recompute warping penalty
            if self.warpreg > 0:
                warp_penalties(X, Y, self._new_penalties)
                self._new_penalties *= self.warpreg
                np.copyto(self._new_losses, self._new_penalties)
            else:
                self._new_losses.fill(0.0)

            # evaluate loss of new warps
            warp_with_quadloss(X, Y, self.template, self._new_losses,
                               self._losses, data)

            # update warping parameters for trials with improved loss
            idx = self._new_losses < self._losses
            self._losses[idx] = self._new_losses[idx]
            self._penalties[idx] = self._new_penalties[idx]
            self.x_knots[idx] = X[idx]
            self.y_knots[idx] = Y[idx]

    def _fit_template(self, data):
        """
        Fit warping template by local random search. Typically, users should
        call either AffineWarping.fit(...) or AffineWarping.continue_fit(...)
        instead of this function.
        """
        K = data.shape[0]
        T = data.shape[1]
        N = data.shape[2]

        if self.l2_smoothness > 0:
            # coefficent matrix for the template update reduce to a
            # banded matrix with 5 diagonals.
            WtW = _diff_gramian(T, self.l2_smoothness * K)
        else:
            # coefficent matrix for the template update reduce to a
            # banded matrix with 3 diagonals.
            WtW = np.zeros((2, T))

        # Compute gramians.
        WtX = np.zeros((T, data.shape[-1]))
        _fast_template_grams(WtW[-2:], WtX, data, self.x_knots, self.y_knots)

        # solve WtW * template = WtX
        self.template = sci.linalg.solveh_banded(WtW, WtX)

        return self.template

    def predict(self):
        """
        Construct model estimate for each trial. Specifically, this warps the
        learned template by the warping function learned for each trial.

        Returns
        -------
        estimate : ndarray, float
            3d array (trials x times x features) holding model reconstruction.
        """
        check_is_fitted(self, 'x_knots')

        # apply warping functions to template
        K = self.x_knots.shape[0]
        T, N = self.template.shape
        result = np.empty((K, T, N))
        return predictwarp(self.x_knots, self.y_knots, self.template, result)

    def argsort_warps(self, t=0.5):
        """Sort trial indices by their warping functions.

        Parameters
        ----------
        t : float
            Fraction of time through the trial. Trials are sorted at this
            timepoint.

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sort trials by magnitude of warping.
        """
        check_is_fitted(self, 'x_knots')
        if t < 0 or t > 1:
            raise ValueError('Test point must be between zero and one.')

        # warp sparse event at t on each trial, and sort the result.
        K = len(self.x_knots)
        y = sparsewarp(self.x_knots, self.y_knots, np.arange(K),
                       np.full(K, t), np.empty(K))
        return np.argsort(y)

    def transform(self, X, return_array=True):
        """
        Apply inverse warping functions to dense or spike data.
        """
        check_is_fitted(self, 'x_knots')

        # get data dimensions
        shape = get_spike_shape(X)
        ndim = len(shape)

        # add append new axis to 2d array if necessary
        if ndim == 2:
            X = X[:, :, None]
        elif ndim != 3:
            raise ValueError('Input should be 2d or 3d array.')

        # check that first axis of X matches n_trials
        if shape[0] != len(self.x_knots):
            raise ValueError('Number of trials in the input does not match '
                             'the number of trials in the fitted model.')

        # length of time axis undergoing warping
        T = shape[1]

        # sparse array transform
        if is_spikedata(X):

            # indices of sparse entries
            trials, times, neurons = get_spike_coords(X)

            # find warped time
            w = sparsealign(self.x_knots, self.y_knots, trials, times / T)

            if return_array:
                # throw away out of bounds spikes
                wtimes = (w * T).astype(int)
                i = (wtimes < T) & (wtimes >= 0)
                return sparse.COO([trials[i], wtimes[i], neurons[i]],
                                  data=np.ones(i.sum()), shape=shape)
            else:
                # return coordinates
                return (trials, w * T, neurons)

        # dense array transform
        else:
            X = np.asarray(X)
            return densewarp(self.y_knots, self.x_knots, X, np.empty_like(X))

    def dump_params(self):
        """
        Returns a list of model parameters for storage
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


@jit(nopython=True)
def _fast_template_grams(WtW, WtX, data, X, Y):

    K, T, N = data.shape
    n_knots = X.shape[1]

    # iterate over trials
    for k in range(len(X)):

        # initialize line segement for interpolation
        y0 = Y[k, 0]
        x0 = X[k, 0]
        slope = (Y[k, 1] - Y[k, 0]) / (X[k, 1] - X[k, 0])

        # 'n' counts knots in piecewise affine warping function.
        n = 1

        # iterate over time bins
        for t in range(T):

            # fraction of trial complete
            x = t / (T - 1)

            # update interpolation point
            while (n < n_knots-1) and (x > X[k, n]):
                y0 = Y[k, n]
                x0 = X[k, n]
                slope = (Y[k, n+1] - y0) / (X[k, n+1] - x0)
                n += 1

            # compute index in warped time
            z = y0 + slope*(x - x0)

            if z >= 1:
                WtX[-1] += data[k, t]
                WtW[1, -1] += 1.0

            elif z <= 0:
                WtX[0] += data[k, t]
                WtW[1, 0] += 1.0

            else:
                i = int(z * (T-1))
                lam = (z * (T-1)) % 1

                WtX[i] += (1-lam) * data[k, t]
                WtW[1, i] += (1-lam)**2
                WtW[1, i+1] += lam**2
                WtW[0, i+1] += (1-lam) * lam
                WtX[i+1] += lam * data[k, t]


@jit(nopython=True)
def _force_monotonic_knots(X, Y):
    K, P = X.shape
    for k in range(K):

        for p in range(P-1):
            x0 = X[k, p]
            y0 = Y[k, p]
            x1 = X[k, p+1]
            y1 = Y[k, p+1]
            dx = X[k, p+1] - X[k, p]
            dy = Y[k, p+1] - Y[k, p]

            if (dx < 0) and (dy < 0):
                # swap both x and y coordinates
                tmp = X[k, p]
                X[k, p] = X[k, p+1]
                X[k, p+1] = tmp
                # swap y
                tmp = Y[k, p]
                Y[k, p] = Y[k, p+1]
                Y[k, p+1] = tmp

            elif dx < 0:
                # swap x coordinate
                tmp = X[k, p]
                X[k, p] = X[k, p+1]
                X[k, p+1] = tmp
                # set y coordinates to mean
                Y[k, p] = Y[k, p] + (dy/2) - 1e-3
                Y[k, p+1] = Y[k, p+1] - (dy/2) + 1e-3

            elif dy < 0:
                # set y coordinates to mean
                Y[k, p] = Y[k, p] + (dy/2) - 1e-3
                Y[k, p+1] = Y[k, p+1] - (dy/2) + 1e-3

        # TODO - redistribute redundant edge knots
        for p in range(P):
            if X[k, p] <= 0:
                X[k, p] = 0.0 + 1e-3*p
            elif X[k, p] >= 1:
                X[k, p] = 1.0 - 1e-3*(P-p-1)

    return X, Y


@jit(nopython=True)
def warp_penalties(X, Y, penalties):

    K = X.shape[0]
    J = X.shape[1]

    for k in range(K):

        # overwrite penalties vector
        penalties[k] = 0

        # left point of line segment
        x0 = X[k, 0]
        y0 = Y[k, 0]

        for j in range(1, J):

            # right point of line segment
            x1 = X[k, j]
            y1 = Y[k, j] - x1  # subtract off identity warp.

            # if y0 and y1 have opposite signs
            if ((y0 < 0) and (y1 > 0)) or ((y0 > 0) and (y1 < 0)):

                # v is the location of the x-intercept expressed as a fraction.
                # v = 1 means that y1 is zero, v = 0 means that y0 is zero
                v = y1 / (y1 - y0)

                # penalty is the area of two right triangles with heights
                # y0 and y1 and bases (x1 - x0) times location of x-intercept.
                penalties[k] += 0.5 * (x1-x0) * ((1-v)*abs(y0) + v*abs(y1))

            # either one of y0 or y1 is zero, or they are both positive or
            # both negative.
            else:

                # penalty is the area of a trapezoid of with height x1 - x0,
                # and with bases y0 and y1
                penalties[k] += 0.5 * abs(y0 + y1) * (x1 - x0)

            # update left point of line segment
            x0 = x1
            y0 = y1

    return penalties


@jit(nopython=True)
def sparsewarp(X, Y, trials, xtst, out):
    """
    Implement inverse warping function at discrete test points, e.g. for
    spike time data.

    Parameters
    ----------
    X : x coordinates of knots for each trial (shape: n_trials x n_knots)
    Y : y coordinates of knots for each trial (shape: n_trials x n_knots)
    trials : int trial id for each coordinate (shape: n_trials)
    xtst : queried x coordinate for each trial (shape: n_trials)

    Note:
        X is assumed to be sorted along axis=1

    Returns
    -------
    ytst : interpolated y value for each x in xtst (shape: trials)
    """

    m = X.shape[0]
    n = X.shape[1]

    for i in range(m):

        if xtst[i] <= 0:
            out[i] = Y[trials[i], 0]

        elif xtst[i] >= 1:
            out[i] = Y[trials[i], -1]

        else:
            x = X[trials[i]]
            y = Y[trials[i]]

            j = 0
            while j < (n-1) and x[j+1] < xtst[i]:
                j += 1

            slope = (y[j+1] - y[j]) / (x[j+1] - x[j])
            out[i] = y[j] + slope*(xtst[i] - x[j])

    return out


def sparsealign(_X, _Y, trials, xtst):
    """
    Parameters
    ----------
    X : x coordinates of knots for each trial (shape: n_trials x n_knots)
    Y : y coordinates of knots for each trial (shape: n_trials x n_knots)
    trials : int trial id for each coordinate (shape: n_trials)
    xtst : queried x coordinate for each trial (shape: n_trials)
    Note:
        X is assumed to be sorted along axis=1
    Returns
    -------
    ytst : interpolated y value for each x in xtst (shape: trials)
    """
    X = _X[trials]
    Y = _Y[trials]

    # allocate result
    ytst = np.empty_like(xtst)

    # for each trial (row of X) find first knot larger than test point
    p = np.argmin(xtst[:, None] > X, axis=1)

    # make sure that we never try to interpolate to the left of
    # X[:,0] to avoid out-of-bounds error. Test points requiring
    # extrapolation are clipped (see below).
    np.maximum(1, p, out=p)

    # indexing vector along trials (used to index with p)
    k = np.arange(len(p))

    # distance between adjacent knots
    dx = np.diff(_X, axis=1)[trials]

    # fractional distance of test points between knots
    lam = (xtst - X[k, p-1]) / dx[k, p-1]

    # linear interpolation
    ytst = (Y[k, p-1]*(1-lam)) + (Y[k, p]*(lam))

    # clip test values below X[:, 0] or above X[:, -1]
    idx = lam > 1
    ytst[idx] = Y[idx, -1]
    idx = lam < 0
    ytst[idx] = Y[idx, 0]

    return ytst


@jit(nopython=True)
def densewarp(X, Y, data, out):

    K = data.shape[0]
    T = data.shape[1]
    n_knots = X.shape[1]

    for k in range(K):

        # initialize line segement for interpolation
        y0 = Y[k, 0]
        x0 = X[k, 0]
        slope = (Y[k, 1] - Y[k, 0]) / (X[k, 1] - X[k, 0])

        # 'n' counts knots in piecewise affine warping function.
        n = 1

        # iterate over all time bins, stop early if loss is too high.
        for t in range(T):

            # fraction of trial complete
            x = t / (T - 1)

            # update interpolation point
            while (n < n_knots-1) and (x > X[k, n]):
                y0 = Y[k, n]
                x0 = X[k, n]
                slope = (Y[k, n+1] - y0) / (X[k, n+1] - x0)
                n += 1

            # compute index in warped time
            z = y0 + slope*(x - x0)

            if z <= 0:
                out[k, t] = data[k, 0]
            elif z >= 1:
                out[k, t] = data[k, -1]
            else:
                _i = z * (T-1)
                rem = _i % 1
                i = int(_i)
                out[k, t] = (1-rem) * data[k, i] + rem * data[k, i+1]

    return out


@jit(nopython=True)
def predictwarp(X, Y, template, out):

    K, T, N = out.shape
    n_knots = X.shape[1]

    for k in range(K):

        # initialize line segement for interpolation
        y0 = Y[k, 0]
        x0 = X[k, 0]
        slope = (Y[k, 1] - Y[k, 0]) / (X[k, 1] - X[k, 0])

        # 'n' counts knots in piecewise affine warping function.
        n = 1

        # iterate over all time bins, stop early if loss is too high.
        for t in range(T):

            # fraction of trial complete
            x = t / (T - 1)

            # update interpolation point
            while (n < n_knots-1) and (x > X[k, n]):
                y0 = Y[k, n]
                x0 = X[k, n]
                slope = (Y[k, n+1] - y0) / (X[k, n+1] - x0)
                n += 1

            # compute index in warped time
            z = y0 + slope*(x - x0)

            if z <= 0:
                out[k, t] = template[0]
            elif z >= 1:
                out[k, t] = template[-1]
            else:
                _i = z * (T-1)
                rem = _i % 1
                i = int(_i)
                out[k, t] = (1-rem) * template[i] + rem * template[i+1]

    return out


@jit(nopython=True)
def warp_with_quadloss(X, Y, template, new_loss, last_loss, data, early_stop=True):

    # num timepoints
    K, T, N = data.shape

    # number discontinuities in piecewise linear function
    n_knots = X.shape[1]

    # normalizing divisor for average loss across each trial
    denom = T * N

    # iterate over trials
    for k in range(K):

        # early stopping
        if early_stop and new_loss[k] >= last_loss[k]:
            break

        # initialize line segement for interpolation
        y0 = Y[k, 0]
        x0 = X[k, 0]
        slope = (Y[k, 1] - Y[k, 0]) / (X[k, 1] - X[k, 0])

        # 'n' counts knots in piecewise affine warping function.
        n = 1

        # iterate over time bins
        for t in range(T):

            # fraction of trial complete
            x = t / (T - 1)

            # update interpolation point
            while (n < n_knots-1) and (x > X[k, n]):
                y0 = Y[k, n]
                x0 = X[k, n]
                slope = (Y[k, n+1] - y0) / (X[k, n+1] - x0)
                n += 1

            # compute index in warped time
            z = y0 + slope*(x - x0)

            # clip warp interpolation between zero and one
            if z <= 0:
                new_loss[k] += _quad_loss(template[0], data[k, t]) / denom

            elif z >= 1:
                new_loss[k] += _quad_loss(template[-1], data[k, t]) / denom

            # do linear interpolation
            else:
                j = z * (T-1)
                rem = j % 1
                new_loss[k] += _interp_quad_loss(
                    rem, template[int(j)], template[int(j)+1], data[k, t]
                ) / denom

            # early stopping
            if early_stop and new_loss[k] >= last_loss[k]:
                break


@jit(nopython=True)
def _quad_loss(pred, targ):
    result = 0.0
    for i in range(pred.size):
        result += (pred[i] - targ[i])**2
    return result


@jit(nopython=True)
def _interp_quad_loss(a, y1, y2, targ):
    result = 0.0
    b = 1 - a
    for i in range(y1.size):
        result += (b*y1[i] + a*y2[i] - targ[i])**2
    return result
