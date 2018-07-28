"""Defines core functionality of Piecewise Linear Time Warping models."""

import numpy as np
import scipy as sci
from numba import jit
from tqdm import trange, tqdm
from sklearn.utils.validation import check_is_fitted

from .spikedata import SpikeData
from .shiftwarp import ShiftWarping
from .utils import _diff_gramian, check_data_tensor


class PiecewiseWarping(object):
    """Piecewise Affine Time Warping applied to an analog (dense) time series.

    Attributes
    ----------
    template : ndarray
        Time series average under piecewise linear warping.
    x_knots : ndarray
        Horizontal coordinates of warping functions.
    y_knots : ndarray
        Vertical coordinates of warping functions.
    loss_hist : list
        History of objective function over optimization.
    """

    def __init__(self, n_knots=0, warp_reg_scale=0, smoothness_reg_scale=0,
                 l2_reg_scale=1e-4, min_temp=-2, max_temp=-1):
        """
        Parameters
        ----------
        n_knots : int
            Nonnegative number specifying number of pieces in the piecewise
            warping function.
        warp_reg_scale : int or float
            Penalty strength on distance of warping functions to identity line.
        smoothness_reg_scale : int or float
            Penalty strength on L2 norm of second temporal derivatives of the
            warping templates.
        l2_reg_scale : int or float
            Penalty strength on L2 norm of the warping template.
        min_temp : int or float
            Smallest mutation rate for evolutionary optimization of warps.
        max_temp : int or float
            Largest mutation rate for evolutionary optimization of warps.
        """

        # check inputs
        if n_knots < 0 or not isinstance(n_knots, int):
            raise ValueError('Number of knots must be nonnegative integer.')

        # model options
        self.n_knots = n_knots
        self.warp_reg_scale = warp_reg_scale
        self.smoothness_reg_scale = smoothness_reg_scale
        self.l2_reg_scale = l2_reg_scale
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.loss_hist = []

    def _mutate_knots(self, temperature):
        """
        Produces a new candidate set of warping functions (as parameterized by
        x and y coordinates of the knots) by randomly perturbing the current
        model parameters.

        Parameters
        ----------
        temperature : scalar
            Scale of perturbation to the knots.

        Returns
        -------
        x, y : ndarray
            Coordinates of new candidate knots defining warping functions.
        """
        K = len(self.x_knots)

        x = self.x_knots + (np.random.randn(K, self.n_knots+2) * temperature)
        x.sort(axis=1)
        x = x - x[:, (0,)]
        x = x / x[:, (-1,)]

        y = self.y_knots + (np.random.randn(K, self.n_knots+2) * temperature)
        y.sort(axis=1)

        return x, y

    def fit(self, data, iterations=10, warp_iterations=20, fit_template=True,
            verbose=True, init_warps='identity', overwrite_loss_hist=True):
        """
        Continues optimization of warps and template (no initialization).

        Parameters
        ----------
        data : ndarray
            3d array (trials x times x features) holding signals to be fit.
        iterations : int
            number of iterations before stopping
        warp_iterations : int
            number of inner iterations to fit warping functions
        verbose (optional) : bool
            whether to display progressbar while fitting (default: True)
        """

        # Initialize warping functions.
        if init_warps == 'identity':
            self.x_knots = np.tile(
                np.linspace(0, 1, self.n_knots+2),
                (data.shape[0], 1)
            )
            self.y_knots = self.x_knots.copy()

        # If 'init_warps' is another warping model, copy the warps from that
        # model.
        elif isinstance(init_warps, (PiecewiseWarping, ShiftWarping)):
            self.copy_fit(init_warps)

        # Check that warps are intialized. If 'init_warps' was not recognized
        # and the warps were not already defined, then raise an exception.
        check_is_fitted(self, ('x_knots', 'y_knots'))

        # Check if warps exist but don't match data dimensions.
        if self.x_knots.shape[0] != data.shape[0]:
            raise ValueError(
                'Initial warping functions must equal the number of trials.'
            )

        # Check input data is provided as a dense array (binned spikes).
        data, is_spikes = check_data_tensor(data)
        if is_spikes:
            raise ValueError("'data' must be provided as a dense numpy array "
                             "(neurons x timepoints x trials) holding binned "
                             "spike data.")

        # Allocate storage for loss.
        K, T, N = data.shape
        self._initialize_storage(K)
        if overwrite_loss_hist:
            self.loss_hist = []

        # Fit model. Alternate between fitting the template and the warping
        # functions.
        pbar = trange(iterations) if verbose else range(iterations)
        for it in pbar:
            self._fit_template(data)
            self._fit_warps(data, warp_iterations)
            self._record_loss(data)

    def _fit_warps(self, data, iterations=20):
        """Fit warping functions by local random search.

        Parameters
        ----------
        data : ndarray
            3d array (trials x times x features) holding time series to be fit.
        iterations : int
            Number of iterations to optimize warps.
        """

        # Decay temperature within each epoch.
        temperatures = np.logspace(self.min_temp, self.max_temp, iterations)

        # Fit warps.
        for temp in reversed(temperatures):

            # Randomly sample warping functions.
            X, Y = self._mutate_knots(temp)

            # Recompute warping penalty.
            if self.warp_reg_scale > 0:
                warp_penalties(X, Y, self._new_penalties)
                self._new_penalties *= self.warp_reg_scale
                np.copyto(self._new_losses, self._new_penalties)
            else:
                self._new_losses.fill(0.0)

            # Evaluate loss of new warps.
            warp_with_quadloss(X, Y, self.template, self._new_losses,
                               self._losses, data)

            # Update warping parameters for trials with improved loss.
            idx = self._new_losses < self._losses
            self._losses[idx] = self._new_losses[idx]
            self._penalties[idx] = self._new_penalties[idx]
            self.x_knots[idx] = X[idx]
            self.y_knots[idx] = Y[idx]

    def _fit_template(self, data):
        """Fit warping template.

        Parameters
        ----------
        data : ndarray
            3d array (trials x times x features) holding time series to be fit.
        """
        K = data.shape[0]
        T = data.shape[1]

        # Initialize gramians based on regularization.
        if self.smoothness_reg_scale > 0:
            WtW = _diff_gramian(
                T, self.smoothness_reg_scale * K, self.l2_reg_scale)
        else:
            WtW = np.zeros((2, T))

        # Compute gramians.
        WtX = np.zeros((T, data.shape[-1]))
        _fast_template_grams(WtW[-2:], WtX, data, self.x_knots, self.y_knots)

        # Solve WtW * template = WtX
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
        return densewarp(self.x_knots, self.y_knots,
                         self.template[None, :, :], result)

    def argsort_warps(self, t=0.5):
        """
        Sort trial indices by their warping functions.

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

    def transform(self, X):
        """
        Apply inverse warping functions to dense or spike data.
        """
        # model must be fitted to perform transform
        check_is_fitted(self, 'x_knots')

        # check that X is an appropriate tensor format
        X, is_spikes = check_data_tensor(X)

        # check that first axis of X matches n_trials
        if X.shape[0] != len(self.x_knots):
            raise ValueError('Number of trials in the input does not match '
                             'the number of trials in the fitted model.')

        # sparse array transform
        if is_spikes:
            # find warped time
            # TODO(ahwillia): replace with sparsewarp
            w = sparsealign(self.x_knots, self.y_knots,
                            X.trials, X.fractional_spiketimes)
            wt = w * (X.tmax - X.tmin) + X.tmin
            return SpikeData(X.trials, wt, X.neurons, X.tmin, X.tmax)

        # dense array transform
        else:
            return densewarp(self.y_knots, self.x_knots, X, np.empty_like(X))

    def event_transform(self, times):
        """
        Time warp events by applying inverse warping functions.
        """

        # model must be fitted to perform transform
        check_is_fitted(self, 'x_knots')

        # check input
        if not isinstance(times, np.ndarray):
            raise ValueError('Input must be a ndarray of event times.')

        # check that there is one event per trial
        if times.shape[0] != len(self.x_knots):
            raise ValueError('Number of trials in the input does not match '
                             'the number of trials in the fitted model.')

        trials = np.arange(times.shape[0])
        # TODO(ahwillia): replace with sparsewarp
        wtimes = sparsealign(self.x_knots, self.y_knots, trials, times)
        return wtimes

    def copy_fit(self, model):
        """
        Copy warping functions and template from another PiecewiseWarping or
        ShiftWarping instance. Useful for warm-starting optimization.
        """

        if isinstance(model, ShiftWarping):
            # check input
            check_is_fitted(model, 'shifts')
            K = len(model.shifts)
            self.x_knots = np.tile(np.linspace(0, 1, self.n_knots+2), (K, 1))
            self.y_knots = self.x_knots - model.fractional_shifts[:, None]
            self.template = model.template.copy()

        elif isinstance(model, PiecewiseWarping):
            # check input
            check_is_fitted(model, 'x_knots')
            if model.n_knots > self.n_knots:
                raise ValueError(
                    "Can't copy fit from another PiecewiseWarping model instance "
                    "with more interior knots."
                )
            # initialize knots
            K = len(model.x_knots)
            rx = np.random.rand(K, self.n_knots - model.n_knots)
            self.x_knots = np.column_stack((model.x_knots, rx))
            self.x_knots.sort(axis=1)
            self.y_knots = np.column_stack([model.event_transform(x) for x in self.x_knots.T])
            self.template = model.template.copy()

        else:
            raise ValueError(
                "Expected either PiecewiseWarping or ShiftWarping model instance."
            )

    def manual_fit(self, data, t0, t1=None, recenter=True):
        """
        Set warping functions and template manually to user-defined events

        Parameters
        ----------
        data : ndarray (trials x time x neurons)
            dense array of neural neural time series data
        t0 : ndarray (trials x 2)
            x position (first column) and y position (second column) of desired
            warping function on each trial
        t1 (optional) : trials x 2
            if specified, x position (first column) and y position (second
            column) of another point in the warping function.

        Notes
        -----
        If t1 is not specified, then the data are transformed only by shifting.
        If t1 is specified, then the data are shifted and scaled to accomdate
        both constraints.
        """

        if self.n_knots > 0:
            raise AttributeError('Manual alignment is only supported for '
                                 'linear warping (n_knots=0) models.')

        # check data dimensions as input
        data, is_spikes = check_data_tensor(data)

        # check if dense array
        if is_spikes:
            raise ValueError("'data' must be provided as a dense numpy array "
                             "(neurons x timepoints x trials) holding binned "
                             "spike data.")

        # check that first warping constraint is well-specified
        if (
            t0.ndim != 2 or
            t0.shape[0] != data.shape[0] or
            t0.shape[1] != 2 or
            not np.issubdtype(t0.dtype, np.floating)
        ):
            raise ValueError("Parameter 't0' must be a K x 2 matrix of "
                             "floating point elements, where K is the number "
                             "of trials in the dataset.")

        # if only one warping constraint is manually specified, use unit slopes
        if t1 is None:
            t1 = t0 + 0.1

        # check that second warping constraint is well-specified
        elif (
            t1.ndim != 2 or
            t1.shape[0] != data.shape[0] or
            t1.shape[1] != 2 or
            not np.issubdtype(t1.dtype, np.floating)
        ):
            raise ValueError("Parameter 't1' must be a K x 2 matrix of "
                             "floating point elements, where K is the number "
                             "of trials in the dataset.")

        # compute slopes and intercepts of the warping functions
        dxdy = (t1 - t0)
        slopes = dxdy[:, 1] / dxdy[:, 0]
        intercepts = t0[:, 1] - slopes * t0[:, 0]

        # recenter warps
        if recenter:
            slopes /= slopes.mean()
            intercepts -= intercepts.mean()

        # use slopes and intercepts to determine warping knots
        self.x_knots = np.tile([0., 1.], (data.shape[0], 1))
        self.y_knots = np.column_stack([intercepts, intercepts + slopes])

        # find best template given these knots and compute model loss
        self._fit_template(data)
        self._losses = np.zeros(data.shape[0])
        self.loss_hist = []
        self._record_loss(data)

    def _initialize_storage(self, n_trials):
        """
        Initializes arrays to hold loss per trial.
        """
        self._losses = np.zeros(n_trials)
        self._penalties = np.zeros(n_trials)
        self._new_losses = np.empty(n_trials)
        self._new_penalties = np.empty(n_trials)

    def _record_loss(self, data):
        """
        Computes overall objective function and appends to self.loss_hist
        """
        # update reconstruction and evaluate loss
        self._losses.fill(0.0)
        warp_with_quadloss(self.x_knots, self.y_knots, self.template,
                           self._losses, self._losses, data, early_stop=False)

        # add warping penalty to losses
        if self.warp_reg_scale > 0:
            self._losses += self._penalties

        # store objective function over time
        self.loss_hist.append(self._losses.mean())


@jit(nopython=True)
def _fast_template_grams(WtW, WtX, data, X, Y):
    """
    Compute Gram matrices for template update least-squares problem.
    """

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
        X and Y are assumed to be sorted along axis=1

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

    K = out.shape[0]
    T = out.shape[1]
    n_knots = X.shape[1]

    for k in range(K):

        # initialize line segement for interpolation
        y0 = Y[k, 0]
        x0 = X[k, 0]
        slope = (Y[k, 1] - Y[k, 0]) / (X[k, 1] - X[k, 0])

        # 'n' counts knots in piecewise affine warping function.
        n = 1

        # broadcast across trials if data has shape 1 x T x N
        if data.shape[0] == 1:
            kk = 0
        else:
            kk = k

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
                out[k, t] = data[kk, 0]
            elif z >= 1:
                out[k, t] = data[kk, -1]
            else:
                _i = z * (T-1)
                rem = _i % 1
                i = int(_i)
                out[k, t] = (1-rem) * data[kk, i] + rem * data[kk, i+1]

    return out


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
                penalties[k] += (0.5 * (x1-x0) * ((1-v)*abs(y0) + v*abs(y1)))**2

            # either one of y0 or y1 is zero, or they are both positive or
            # both negative.
            else:

                # penalty is the area of a trapezoid of with height x1 - x0,
                # and with bases y0 and y1
                penalties[k] += (0.5 * abs(y0 + y1) * (x1 - x0))**2

            # update left point of line segment
            x0 = x1
            y0 = y1

    return penalties
