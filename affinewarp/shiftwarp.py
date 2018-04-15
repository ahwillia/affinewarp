import numpy as np
from scipy.signal import correlate


class ShiftWarping(object):
    """Represents a collection of time series, each with an affine time warp.
    """
    def __init__(self, data, maxlag=.1):
        """
        Params
        ------
        data (ndarray) : n_trials x n_timepoints x n_features
        """

        # data dimensions
        self.data = data
        self.n_trials = data.shape[0]
        self.n_timepoints = data.shape[1]
        self.n_features = data.shape[2]
        self.maxlag = maxlag
        self.L = int(self.n_timepoints * maxlag)

        self.template = data.mean(axis=0)
        self.loss_hist = []
        self.shifts = np.ones(self.n_trials, dtype=np.int32)

    def fit(self, data, iterations=10):

        losses = np.empty((K, 2*L+1))
        template = data.mean(axis=0)

        pbar = trange(iterations)
        self.loss_hist = []

        for i in pbar:

            # compute the loss for each shift
            losses.fill(0.0)
            _compute_shifted_loss(data, template, losses)

            # find the best shift for each trial
            s = np.argmax(losses, axis=1)
            self.shifts = s - L

            # compute the total loss
            total_loss = np.sqrt(losses[rk, s].sum())
            self.loss_hist.append(total_loss)

            pbar.set_description('Loss: {0:.2f}'.format(total_loss))

    def predict(self):




def _compute_shifted_loss(data, template, losses):

    L = losses.shape[1] // 2

    for k in range(data.shape[0]):
        for t in range(data.shape[1]):
            for l in range(-L, L+1):
                for n in range(data.shape[2]):
                    loss[k, l] += (data[k, t, n] - template[k, t-l, n])**2
