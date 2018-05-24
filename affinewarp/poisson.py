import numpy as np
import scipy as sci
from tqdm import trange, tqdm
from .utils import _diff_gramian
from sklearn.utils.validation import check_is_fitted
from .spikedata import get_spike_coords, get_spike_shape, is_spikedata
from .spikedata import assert_spike_data, trial_average_spikes
from numba import jit
import sparse


class PoissonAffineWarp(object):
    """Piecewise Affine Time Warping applied to an analog (dense) time series.
    """
    def __init__(self, n_knots=0, warpreg=0, l2_smoothness=0,
                 nbins=100, min_temp=-2, max_temp=-1):

        # check inputs
        if n_knots < 0:
            raise ValueError('Number of knots must be nonnegative.')

        # model options
        self.n_knots = n_knots
        self.warpreg = warpreg
        self.l2_smoothness = l2_smoothness
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.nbins = nbins

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
        return force_monotonic_knots(x, y)

    def fit(self, data, **kwargs):
        """
        Initializes warping functions and model template and begin fitting.
        """
        assert_spike_data(data)
        K, T, N = get_spike_shape(data)
        trials, times, neurons = get_spike_coords(data)

        # initialize template
        self.template = trial_average_spikes(data, self.nbins)

        # initialize warping functions to identity
        self.x_knots = np.tile(
            np.linspace(0, 1, self.n_knots+2),
            (K, 1)
        )
        self.y_knots = self.x_knots.copy()

        # compute initial loss
        self._losses = np.zeros(K)
        warp_with_poissloss(self.x_knots, self.y_knots, self.template,
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
                waro_with_poissloss(self.x_knots, self.y_knots, self.template,
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

        trials, times, neurons = get_spike_coords(data)

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
            warp_with_poissloss(X, Y, self.predict(), self._new_losses,
                                trials, times, neurons)

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

        # NOTE -- need to optimize over log-firing rates

        # TODO -- incorporate smoothness penalty

        # optimize over the log firing rates
        log_temp = np.log(self.template)
        pred = self.predict()

        return self.template


@jit(nopython=True)
def warp_with_poissloss(X, Y, pred, loss, trials, times, neurons):

    # iterate over each spike.
    #   - k is the trial number for each spike
    #   - t is the spike time between zero and one
    #   - n is the id of the neuron which fired the spike
    for k, t, n in zip(trials, times, neurons):

        # floor and ceil of timed index
        tf = int(t * (T-1))
        tc = tf + 1

        # find estimated firing rate for this spike
        if tf == (T-1):
            # if twarped == 1, model estimate is final
            f = pred[k, tf, n]

        else:
            # linear interpolation between template bins
            f = (tc - t)*pred[k, tf, n] + (t - tf)*pred[k, tc, n]

        # add contribution of spike to loss
        loss[k] -= np.log(f)


@jit(nopython=True)
def template_gradient(X, Y, pred, trials, times, neurons, grad_out):
    """
    pred : warped firing rates
    """

    # iterate over each spike.
    #   - k is the trial number for each spike
    #   - t is the spike time between zero and one
    #   - n is the id of the neuron which fired the spike
    for k, t, n in zip(trials, times, neurons):

        # floor and ceil of timed index
        tf = int(t * (T-1))
        tc = tf + 1

        # find estimated firing rate for this spike
        if tf == (T-1):
            # if twarped == 1, model estimate is the final bin
            f = pred[k, tf, n]
            grad_out[tf, n] += 1 / f

        else:
            alpha = tc - t  # closeness to tf
            beta = t - tf   # closeness to tc

            # model estimated firing rate for this spike
            f = alpha*pred[k, tf, n] + beta*pred[k, tc, n]

            grad_out[tf, n] -= alpha / f
            grad_out[tc, n] -= beta / f

        # add contribution to loss
        loss -= np.log(f)
