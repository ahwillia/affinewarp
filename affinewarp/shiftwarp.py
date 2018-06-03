import numpy as np
from numba import jit
from tqdm import trange
import scipy as sci
from sklearn.utils.validation import check_is_fitted
from .spikedata import is_spike_data, get_spike_shape, get_spike_coords
from .utils import _diff_gramian, check_data_tensor
import sparse


class ShiftWarping(object):
    """Represents a collection of time series, each with an affine time warp.
    """
    def __init__(self, maxlag=.2, warpreg=0, l2_smoothness=0):
        """
        Params
        ------
        maxlag : float
            maximal allowable shift
        warpreg : float
            strength of penalty on the magnitude of the shifts
        l2_smoothness : float
            strength of roughness penalty on the template
        """

        # data dimensions
        self.maxlag = maxlag
        self.loss_hist = []
        self.warpreg = warpreg
        self.l2_smoothness = l2_smoothness

    def fit(self, data, iterations=10, verbose=True):
        """Fit shift warping to data.
        """

        # data dimensions:
        #   K = number of trials
        #   T = number of timepoints
        #   N = number of features/neurons
        K, T, N = data.shape

        # initialize
        DtD = _diff_gramian(T, self.l2_smoothness * K)
        self.shifts = np.zeros(K, dtype=int)

        L = int(self.maxlag * T)
        losses = np.empty((K, 2*L+1))
        self.loss_hist = []

        WtW = np.zeros((3, T))
        WtX = np.zeros((T, N))
        _fill_WtW(self.shifts, WtW[-1])
        _fill_WtX(data, self.shifts, WtX)
        self.template = sci.linalg.solveh_banded((WtW + DtD), WtX)

        pbar = trange(iterations) if verbose else range(iterations)


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

        self.fractional_shifts = self.shifts / T

    def argsort_warps(self):
        check_is_fitted(self, 'shifts')
        return np.argsort(self.shifts)

    def predict(self):
        check_is_fitted(self, 'shifts')
        K = len(self.shifts)
        T, N = self.template.shape
        pred = np.empty((K, T, N))
        _predict(self.template, self.shifts, pred)
        return pred

    def transform(self, data):
        check_is_fitted(self, 'shifts')
        data, is_spikes = check_data_tensor(data)

        if is_spikes:
            # indices of sparse entries
            shape = get_spike_shape(data)
            trials, times, neurons = get_spike_coords(data)
            T = shape[1]
            wtimes = (((times/T) - self.fractional_shifts[trials])*T).astype(int)
            i = (wtimes > 0) & (wtimes < T)
            return sparse.COO([trials[i], wtimes[i], neurons[i]],
                              data=np.ones(i.sum()), shape=shape)

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
