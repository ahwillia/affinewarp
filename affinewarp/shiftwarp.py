import numpy as np
from numba import jit
from tqdm import trange


class ShiftWarping(object):
    """Represents a collection of time series, each with an affine time warp.
    """
    def __init__(self, maxlag=.2, shift_penalty=.1):
        """
        Params
        ------
        """

        # data dimensions
        self.maxlag = maxlag
        self.template = None
        self.loss_hist = []
        self.shift_penalty = .1

    def fit(self, data, iterations=10, verbose=True):

        # data dimensions:
        #   K = number of trials
        #   T = number of timepoints
        #   N = number of features/neurons
        K, T, N = data.shape

        L = int(self.maxlag * T)
        losses = np.empty((K, 2*L+1))
        self.WtW = np.zeros(T)
        self.WtX = np.zeros((T, N))
        self.template = data.mean(axis=0)
        # regularization = self.shift_penalty *\
        #     np.tile(np.abs(np.arange(-L, L+1)), (K, 1))

        pbar = trange(iterations) if verbose else range(iterations)
        self.loss_hist = []

        for i in pbar:

            # compute the loss for each shift
            losses.fill(0.0)
            _compute_shifted_loss(data, self.template, losses)

            # find the best shift for each trial
            s = np.argmin(losses, axis=1)
            self.shifts = -L + s

            # compute the total loss
            total_loss = np.sqrt(losses[np.arange(K), s].sum())
            self.loss_hist.append(total_loss)

            # update loss display
            if verbose:
                pbar.set_description('Loss: {0:.2f}'.format(total_loss))

            # update template
            self.WtW.fill(0.0)
            self.WtX.fill(0.0)

            _fill_WtW(self.shifts, self.WtW)
            _fill_WtX(data, self.shifts, self.WtX)

            self.template = self.WtX / self.WtW[:, None]

    def predict(self):
        K = len(self.shifts)
        T, N = self.template.shape
        pred = np.empty((K, T, N))
        _predict(self.template, self.shifts, pred)
        return pred

    def transform(self, data):
        K, T, N = data.shape
        out = np.empty_like(data)
        _warp_data(data, self.shifts, out)
        return out


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
        if s <= 0:
            out[0] += 1 - s
            for i in range(1, T + s):
                out[i] += 1
        elif s > 0:
            out[-1] += 1 + s
            for i in range(2, T - s + 1):
                out[-i] += 1


@jit(nopython=True)
def _fill_WtX(data, shifts, out):
    K, T, N = data.shape
    for k in range(K):
        i = -shifts[k]
        t = 0
        while i < 0:
            out[0] += data[k, t]
            t += 1
            i += 1
        while (i < T) and (t < T):
            out[i] += data[k, t]
            t += 1
            i += 1
        while t < T:
            out[-1] += data[k, t]
            t += 1


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
                for n in range(N):
                    i = t - l
                    if i < 0:
                        i = 0
                    elif i >= T:
                        i = T-1
                    losses[k, l+L] += (data[k, t, n] - template[i, n])**2
