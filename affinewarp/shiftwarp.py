import numpy as np
from numba import jit
from tqdm import trange
import scipy as sci
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy

from .spikedata import SpikeData
from .utils import _diff_gramian, check_data_tensor


class ShiftWarping(object):
    """
    Models translations in time across a collection of multi-dimensional time
    series. Does not model stretching or compression of time across trials.

    Attributes
    ----------
    shifts : ndarray
        Number of time bins shifted on each trial.
    fractional_shifts : ndarray
        Time shifts expressed as a fraction of trial length.
    loss_hist : list
        History of objective function over optimization.
    """

    def __init__(self, maxlag=.5, warp_reg_scale=0, smoothness_reg_scale=0,
                 l2_reg_scale=1e-4):
        """Initializes ShiftWarping object with hyperparameters.

        Parameters
        ----------
        maxlag : float
            Maximal allowable shift.
        warp_reg_scale : float
            Penalty strength on the magnitude of the shifts.
        smoothness_reg_scale : int or float
            Penalty strength on L2 norm of second temporal derivatives of the
            warping templates.
        l2_reg_scale : int or float
            Penalty strength on L2 norm of the warping template.
        """

        if (maxlag < 0) or (maxlag > .5):
            raise ValueError('Value of maxlag must be between 0 and 0.5')

        self.maxlag = maxlag
        self.loss_hist = []
        self.warp_reg_scale = warp_reg_scale
        self.smoothness_reg_scale = smoothness_reg_scale
        self.l2_reg_scale = l2_reg_scale

    def fit(self, data, iterations=10, verbose=True, warp_iterations=None):
        """
        Fit shift warping to data.
        """

        # TODO - support this?
        if isinstance(data, SpikeData):
            raise NotImplementedError()

        # data dimensions:
        #   K = number of trials
        #   T = number of timepoints
        #   N = number of features/neurons
        K, T, N = data.shape

        # initialize shifts
        self.shifts = np.zeros(K, dtype=int)
        L = int(self.maxlag * T)

        # compute gramian for regularization term
        DtD = _diff_gramian(
            T, self.smoothness_reg_scale * K, self.l2_reg_scale)

        # initialize template
        WtW = np.zeros((3, T))
        WtX = np.zeros((T, N))
        _fill_WtW(self.shifts, WtW[-1])
        _fill_WtX(data, self.shifts, WtX)
        self.template = sci.linalg.solveh_banded((WtW + DtD), WtX)

        # initialize learning curve
        losses = np.empty((K, 2*L+1))
        self.loss_hist = []
        pbar = trange(iterations) if verbose else range(iterations)

        # main loop
        for i in pbar:

            # compute the loss for each shift
            losses.fill(0.0)
            _compute_shifted_loss(data, self.template, losses)
            losses /= (T * N)

            # find the best shift for each trial
            s = np.argmin(losses, axis=1)

            self.shifts = -L + s

            # compute the total loss
            total_loss = np.mean(losses[np.arange(K), s])
            self.loss_hist.append(total_loss)

            # update loss display
            if verbose:
                pbar.set_description('Loss: {0:.2f}'.format(total_loss))

            # update template
            WtW.fill(0.0)
            WtX.fill(0.0)

            _fill_WtW(self.shifts, WtW[-1])
            _fill_WtX(data, self.shifts, WtX)

            self.template = sci.linalg.solveh_banded((WtW + DtD), WtX)

        # compute shifts as a fraction of trial length
        self.fractional_shifts = self.shifts / T

    def argsort_warps(self):
        """
        Returns an ordering of the trials based on the learned shifts.
        """
        check_is_fitted(self, 'shifts')
        return np.argsort(self.shifts)

    def predict(self):
        """
        Returns model prediction (warped version of template on each trial).
        """
        check_is_fitted(self, 'shifts')

        # Allocate space for prediction.
        K = len(self.shifts)
        T, N = self.template.shape
        pred = np.empty((K, T, N))

        # Compute prediction in JIT-compiled function.
        _predict(self.template, self.shifts, pred)
        return pred

    def transform(self, data):
        """
        Applies inverse warping functions to align raw data across trials.
        """
        check_is_fitted(self, 'shifts')
        data, is_spikes = check_data_tensor(data)

        # For SpikeData objects
        if is_spikes:
            d = deepcopy(data)
            return d.shift_each_trial_by_fraction(self.fractional_shifts)

        else:
            # warp dense data
            K, T, N = data.shape
            out = np.empty_like(data)
            _warp_data(data, self.shifts, out)
            return out

    def event_transform(self, times):
        # must be fitted before transform
        check_is_fitted(self, 'shifts')

        # check input
        if not isinstance(times, np.ndarray):
            raise ValueError('Input must be a ndarray of event times.')

        # check that there is one event per trial
        if times.shape[0] != len(self.shifts):
            raise ValueError('Number of trials in the input does not match '
                             'the number of trials in the fitted model.')

        return times - self.fractional_shifts


@jit(nopython=True)
def _predict(template, shifts, out):
    """
    Produces model prediction. Applies shifts to template on each trial.

    Parameters
    ----------
    template : array_like
        Warping template, shape: (time x features)
    shifts : array_like
        Learned shifts on each trial, shape: (trials)
    out : array_like
        Storage for model prediction, shape: (trials x time x features)

    Notes
    -----
    This private function is just-in-time compiled by numba. See
    ShiftWarping.predict(...) wraps this function.
    """
    K = len(shifts)
    T, N = template.shape
    for k in range(K):
        i = -shifts[k]
        t = 0
        while i < 0:
            out[k, t] = template[0]
            t += 1
            i += 1
        while (i < T) and (t < T):
            out[k, t] = template[i]
            t += 1
            i += 1
        while t < T:
            out[k, t] = template[-1]
            t += 1


@jit(nopython=True)
def _fill_WtW(shifts, out):
    T = len(out)
    for s in shifts:
        if s < 0:
            out[-1] += 1 - s
            for i in range(2, T + s + 1):
                out[-i] += 1
        else:
            out[0] += 1 + s
            for i in range(1, T - s):
                out[i] += 1


@jit(nopython=True)
def _fill_WtX(data, shifts, out):
    K, T, N = data.shape
    for k in range(K):
        i = -shifts[k]
        t = 0

        for t in range(T):
            if i < 0:
                out[0] += data[k, t]
            elif i >= T:
                out[-1] += data[k, t]
            else:
                out[i] += data[k, t]
            i += 1


@jit(nopython=True)
def _warp_data(data, shifts, out):
    K, T, N = data.shape
    for k in range(K):
        i = shifts[k]
        t = 0
        while i < 0:
            out[k, t] = data[k, 0]
            t += 1
            i += 1
        while (i < T) and (t < T):
            out[k, t] = data[k, i]
            t += 1
            i += 1
        while t < T:
            out[k, t] = data[k, -1]
            t += 1


@jit(nopython=True)
def _compute_shifted_loss(data, template, losses):

    K, T, N = data.shape
    L = losses.shape[1] // 2

    for k in range(K):
        for t in range(T):
            for l in range(-L, L+1):

                # shifted index
                i = t - l
                if i < 0:
                    i = 0
                elif i >= T:
                    i = T-1

                # quadratic loss
                for n in range(N):
                    losses[k, l+L] += (data[k, t, n] - template[i, n])**2
