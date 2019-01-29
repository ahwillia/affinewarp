import numpy as np
import numba
from tqdm import trange
import scipy as sci
from sklearn.utils.validation import check_is_fitted
import scipy.optimize

from .spikedata import SpikeData
from .utils import check_dimensions
from ._optimizers import _diff_gramian, PoissonObjective


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
                 l2_reg_scale=1e-7, loss='quadratic'):
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
        self.loss = loss

        if loss == 'quadratic':
            self._shifted_loss = _compute_shifted_quad_loss
        elif loss == 'poisson':
            self._shifted_loss = _compute_shifted_poiss_loss

    def fit(self, data, iterations=20, verbose=True, warp_iterations=None):
        """
        Fit shift warping to data.
        """

        # Check input.
        if isinstance(data, SpikeData):
            raise ValueError("Expected input 'data' to be a 3d numpy "
                             "holding binned spike counts.")

        # data dimensions:
        #   K = number of trials
        #   T = number of timepoints
        #   N = number of features/neurons
        #   L = number of time lags/shifts to search over in each direction.
        K, T, N = data.shape
        L = int(self.maxlag * T)

        # Warps are penalized based on distance from identity. These penalties
        # can be pre-computed up front.
        warp_penalty = self.warp_reg_scale * \
            np.abs(np.linspace(-L/T, L/T, 2*L+1)[None, :])

        # Initialize shifts.
        self.shifts = np.zeros(K, dtype=int)

        # Initialize model template.
        self._fit_template(data)

        # Compute the model loss over all shifts.
        losses = np.zeros((K, 2*L+1))
        self._shifted_loss(data, self.template, losses)
        losses /= (T * N)

        # Initialize learning curve with zero warping.
        self.loss_hist = [losses[:, L].mean()]

        # progress bar
        pbar = trange(iterations) if verbose else range(iterations)

        # main loop
        for i in pbar:

            # Find the best shift for each trial.
            s = np.argmin(losses + warp_penalty, axis=1)
            self.shifts = -L + s

            # Compute the loss after shifting
            total_loss = np.mean(losses[np.arange(K), s])
            self.loss_hist.append(total_loss)

            # update loss display
            if verbose:
                pbar.set_description('Loss: {0:.2f}'.format(total_loss))

            # Re-fit model template and re-compute losses over all shifts.
            if (i + 1) < iterations:
                self._fit_template(data)
                losses.fill(0.0)
                self._shifted_loss(data, self.template, losses)
                losses /= (T * N)

        # compute shifts as a fraction of trial length
        self.fractional_shifts = self.shifts / T

    def _fit_template(self, data):
        K, T, N = data.shape

        if self.loss == 'quadratic':
            DtD = _diff_gramian(T, self.smoothness_reg_scale * K, self.l2_reg_scale *K)
            WtW = np.zeros((3, T))
            WtX = np.zeros((T, N))
            _fill_WtW(self.shifts, WtW[-1])
            _fill_WtX(data, self.shifts, WtX)
            self.template = sci.linalg.solveh_banded((WtW + DtD), WtX)

        elif self.loss == 'poisson':
            # Get initial parameters.
            try:
                x0 = self.template.ravel()
            except AttributeError:
                x0 = np.zeros(T * N)

            # Set up optimization problem.
            obj = PoissonObjective(data, self.smoothness_reg_scale,
                                   self.l2_reg_scale, shifts=self.shifts)

            # Fit template.
            opt = scipy.optimize.minimize(obj, x0,
                                          jac=True, hessp=obj.hessp,
                                          method='newton-cg',
                                          options=dict(maxiter=1))
            self.template = (opt.x).reshape(T, N)

    def argsort_warps(self):
        """
        Returns an ordering of the trials based on the learned shifts.
        """
        self.assert_fitted()
        return np.argsort(self.shifts)

    def predict(self):
        """
        Returns model prediction (warped version of template on each trial).
        """
        self.assert_fitted()

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
        self.assert_fitted()
        data, is_spikes = check_dimensions(self, data)

        # For SpikeData objects.
        if is_spikes:
            return data.shift_each_trial_by_fraction(self.fractional_shifts)

        # For dense data (trials x timebins x units).
        else:
            # warp dense data
            out = np.empty_like(data)
            _warp_data(data, self.shifts, out)
            return out

    def event_transform(self, trials, frac_times):
        """
        Time warp events by applying inverse warping functions.

        Parameters
        ----------
        trials : array-like
            Vector of ints holding trial indices for each event.
        frac_times : array-like
            Vector of floats holding fractional times for each event
            (``frac_times[i] == 0`` means that event ``i`` occured at
            trial start; ``frac_times[i] == 1`` means that event ``i``
            occured at trial end).

        Returns
        -------
        aligned_times : array-like
            Transformed fractional event times.
        """
        self.assert_fitted()
        return np.asarray(frac_times)[trials] - self.fractional_shifts[trials]

    def copy_fit(self, model):
        """
        Applies inverse warping functions to align raw data across trials.
        """
        if not isinstance(model, ShiftWarping):
            raise ValueError("Can only copy another ShiftWarping instance.")
        model.assert_fitted()
        self.template = model.template.copy()
        self.shifts = model.shifts
        self.fractional_shifts = model.fractional_shifts
        return self

    def assert_fitted(self):
        check_is_fitted(self, 'shifts')


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True, parallel=True)
def _compute_shifted_quad_loss(data, template, losses):

    K, T, N = data.shape
    L = losses.shape[1] // 2

    for k in numba.prange(K):
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


@numba.jit(nopython=True, parallel=True)
def _compute_shifted_poiss_loss(data, template, losses):

    K, T, N = data.shape
    L = losses.shape[1] // 2

    exp_template = np.exp(template)

    for k in numba.prange(K):
        for t in range(T):
            for l in range(-L, L+1):

                # shifted index
                i = t - l
                if i < 0:
                    i = 0
                elif i >= T:
                    i = T-1

                # poisson loss
                for n in range(N):
                    losses[k, l+L] += exp_template[i, n] - template[i, n] * data[k, t, n]
