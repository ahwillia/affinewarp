"""Defines core functionality of Piecewise Linear Time Warping models."""

import numpy as np
import numba
import numbers
from tqdm import trange, tqdm
from sklearn.utils.validation import check_is_fitted

from .spikedata import SpikeData
from .shiftwarp import ShiftWarping
from .utils import check_dimensions

from ._optimizers import OptimizerFactory, warp_penalties
optimizer_factory = OptimizerFactory()

_DATA_ERROR = ValueError("'data' must be provided as a dense numpy array "
                         "(neurons x timepoints x trials) holding binned "
                         "spike data.")


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

    def __init__(self, n_knots=0, warp_reg_scale=0.0, smoothness_reg_scale=0.0,
                 l2_reg_scale=1e-7, min_temp=-3, max_temp=-1.5, n_restarts=1,
                 loss='quadratic'):
        """
        Parameters
        ----------
        n_knots : int, default 0
            Nonnegative number specifying number of pieces in the piecewise
            warping function.
        warp_reg_scale : int or float, default 0.0
            Penalty strength on distance of warping functions to identity line.
        smoothness_reg_scale : int or float, default 0.0
            Penalty strength on L2 norm of second temporal derivatives of the
            warping templates.
        l2_reg_scale : int or float, default 1e-4
            Penalty strength on L2 norm of the warping template.
        min_temp : int or float, default -3
            Smallest mutation rate for evolutionary optimization of warps.
        max_temp : int or float, default -1.5
            Largest mutation rate for evolutionary optimization of warps.
        n_restarts : int, default 1
            Number of times to restart optimization on warps.
        """

        # check inputs
        if n_knots < -1 or not isinstance(n_knots, numbers.Integral):
            raise ValueError('Number of knots must be an integer >= -1.')

        # model options
        self.n_knots = n_knots
        self.warp_reg_scale = warp_reg_scale
        self.smoothness_reg_scale = smoothness_reg_scale
        self.l2_reg_scale = l2_reg_scale
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.n_restarts = n_restarts
        self.template = None
        self.loss_hist = []

        self.loss = loss
        funcs = optimizer_factory(loss)
        self._template_optimizer = funcs[0]
        self._warp_optimizer = funcs[1]
        self._eval_loss = funcs[2]

    def initialize_warps(self, n_trials, init_warps=None):
        """
        Initializes warping functions. Sets ``self.x_knots`` and
        ``self.y_knots`` to identity by default, or copies them from a
        different model (see ``self.copy_fit``).

        Parameters
        ----------
        n_trials : int
            Number of trials.
        init_warps (optional) : PiecewiseWarping instance, ShiftWarping instance, or None.
            Model specifying initial warps. If None, warps are initialized to
            the identity line. (Default: None)
        """

        if init_warps is None:
            # Initialize warps to identity.
            self.x_knots = np.tile(
                np.linspace(0, 1, max(2, self.n_knots + 2)),
                (n_trials, 1)
            )
            self.y_knots = self.x_knots.copy()
        elif isinstance(init_warps, (PiecewiseWarping, ShiftWarping)):
            # Copy warps from another model
            self.copy_fit(init_warps)
        else:
            raise ValueError("Parameter 'init_warps' misspecified. Expected "
                             "a PiecewiseWarping or ShiftWarping instance.")

        # Check if initial warps don't match data dimensions.
        if self.x_knots.shape[0] != n_trials:
            raise ValueError("Initial warping functions must equal the number "
                             "of trials.")

    def fit(self, data, iterations=50, warp_iterations=200, verbose=True,
            init_warps=None, trial_idx=slice(None), neuron_idx=slice(None)):
        """
        Fits warping functions and model template to data.

        Parameters
        ----------
        data : ndarray
            3d array (trials x times x features) holding signals to be fit.
        iterations (optional) : int
            Number of iterations before stopping.
        warp_iterations (optional) : int
            Number of inner iterations to fit warping functions.
        verbose (optional) : bool
            Whether to display progressbar while fitting. (Default: True)
        init_warps (optional) : PiecewiseWarping instance, ShiftWarping
            instance, or None. Model specifying initial warps. If None, warps
            are initialized to the identity line. (Default: None)
        trial_idx (optional) : indices
            Indices along first dimension of `data` array. If provided, the
            warping template is only fit to these trials. By default, all
            trials are used for fitting.
        neuron_idx (optional) : indices
            Indices along last dimension of `data` array. If provided, the
            warping functions are only fit to these units. By default, all
            units are used for fitting.
        """

        # Check input data is provided as a dense array (binned spikes).
        if not isinstance(data, np.ndarray):
            raise _DATA_ERROR

        # Check number of dimensions.
        data = data[:, :, None] if data.ndim == 2 else data
        if data.ndim != 3:
            raise _DATA_ERROR

        # Allocate storage.
        K, T, N = data.shape
        self.initialize_warps(K, init_warps)
        self._initialize_storage(K)

        # Ensure template is initialized.
        self._fit_template(data, trial_idx)

        # Record initial loss before optimizing.
        self._record_loss(data)

        # Fit model. Alternate between fitting the template and the warping
        # functions.
        pbar = trange(iterations) if verbose else range(iterations)
        self._knot_hist = []
        for it in pbar:
            self._fit_warps(data, warp_iterations, neuron_idx)
            self._fit_template(data, trial_idx)
            self._record_loss(data)
            if verbose:
                raw_imp = (self.loss_hist[0] - self.loss_hist[-1])
                rel_imp = 100 * raw_imp / self.loss_hist[0]
                pbar.set_description(
                    "Loss improvement: {0:.2f}%".format(rel_imp))

        return self

    def _fit_warps(self, data, warp_iterations, neuron_idx):
        """Fit warping functions by local random search.

        Parameters
        ----------
        data : ndarray
            3d array (trials x times x features) holding time series to be fit.
        warp_iterations : int
            Number of iterations to optimize warps.
        neuron_idx : indices
            Specifies subset of neurons to fit warps on.
        """
        storage = np.empty((data.shape[0], 4, max(2, self.n_knots + 2)))
        is_shift_only = self.n_knots < 0
        self._warp_optimizer(
            self.x_knots, self.y_knots, self.template[:, neuron_idx],
            data[:, :, neuron_idx], self.warp_reg_scale, self._losses,
            self._penalties, warp_iterations, self.n_restarts,
            self.min_temp, self.max_temp, storage, is_shift_only)

    def _fit_template(self, data, trial_idx=slice(None)):
        """Fit warping template.

        Parameters
        ----------
        data : ndarray
            3d array (trials x times x features) holding time series to be fit.
        trial_idx : indices
            Specifies subset of trials to fit template on.
        """
        self.template = self._template_optimizer(
            self.x_knots[trial_idx], self.y_knots[trial_idx], self.template,
            data[trial_idx], self.smoothness_reg_scale, self.l2_reg_scale)

    def predict(self):
        """
        Construct model estimate for each trial. Specifically, this warps the
        learned template by the warping function learned for each trial.

        Returns
        -------
        estimate : ndarray, float
            3d array (trials x times x features) holding model reconstruction.
        """
        self.assert_fitted()

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
        self.assert_fitted()
        if t < 0 or t > 1:
            raise ValueError('Test point must be between zero and one.')

        # warp sparse event at t on each trial, and sort the result.
        K = len(self.x_knots)
        y = sparsewarp(self.x_knots, self.y_knots, np.arange(K),
                       np.full(K, t), np.empty(K))
        return np.argsort(y)

    def transform(self, data):
        """
        Apply inverse warping functions to dense or spike data.

        Parameters
        ----------
        data : ndarray or SpikeData instance
            Time series data to be transformed.

        Returns
        -------
        aligned_data
        """
        # Check that model is fitted. Check dimensions and rename data -> X.
        self.assert_fitted()
        X, is_spikes = check_dimensions(self, data)

        # Transform spike train.
        if is_spikes:
            w = sparsewarp(self.x_knots, self.y_knots, X.trials,
                           X.fractional_spiketimes, np.empty(X.n_spikes))
            wt = w * (X.tmax - X.tmin) + X.tmin
            return SpikeData(X.trials, wt, X.neurons, X.tmin, X.tmax)

        # Transform dense data array (trials x timebins x units).
        else:
            return densewarp(self.y_knots, self.x_knots, X, np.empty_like(X))

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

        # Check that model is fitted and inputs have appropriate dimensions.
        self.assert_fitted()
        trials = np.squeeze(np.asarray(trials))
        frac_times = np.squeeze(np.asarray(frac_times))

        if (trials.ndim != 1) or (frac_times.ndim != 1):
            raise ValueError("Expected inputs to be 1D sequences.")

        if not np.issubdtype(trials.dtype, np.integer):
            raise ValueError("Expected 'trials' to be an array containing "
                             "integer indices.")

        if len(trials) != len(frac_times):
            raise ValueError("Expected 'trials' and 'frac_times' to have "
                             "equal lengths.")

        if trials.min() < 0:
            raise ValueError("Expected 'trials' to contain nonnegative "
                             "integer indices, but saw negative entries.")

        if trials.max() >= self.x_knots.shape[0]:
            raise ValueError("Dimension mismatch. Model was fit on a dataset "
                             "containing {} trials, but input 'trials' had "
                             "indices larger than "
                             "this.".format(self.x_knots.shape[0]))

        return sparsewarp(self.x_knots, self.y_knots, trials,
                          frac_times, np.empty_like(frac_times))

    def copy_fit(self, model):
        """
        Copy warping functions and template from another PiecewiseWarping or
        ShiftWarping instance. Useful for warm-starting optimization.
        """

        if isinstance(model, ShiftWarping):
            model.assert_fitted()
            K = len(model.shifts)
            self.x_knots = np.tile(np.linspace(0, 1, self.n_knots+2), (K, 1))
            self.y_knots = self.x_knots - model.fractional_shifts[:, None]
            self.template = model.template.copy()

        elif isinstance(model, PiecewiseWarping):
            # Check input.
            model.assert_fitted()
            if model.n_knots > self.n_knots:
                raise ValueError(
                    "Can't copy fit from another PiecewiseWarping model "
                    "instance with more interior knots."
                )
            K = len(model.x_knots)

            # Initialize x knots, if self has additional knots place them
            # randomly at random positions.
            rx = np.random.rand(K, self.n_knots - model.n_knots)
            self.x_knots = np.column_stack((model.x_knots, rx))
            self.x_knots.sort(axis=1)

            # Place y knots so that warping function matches copied model.
            # Additional / redundant knots are placed randomly.
            y = [model.event_transform(np.arange(len(x)), x) for x in self.x_knots.T]
            self.y_knots = np.column_stack(y)

            # Copy template.
            self.template = model.template.copy()

        else:
            raise ValueError(
                "Expected either PiecewiseWarping or ShiftWarping model instance."
            )

        return self

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

        # check if dense array
        if not isinstance(data, np.ndarray):
            raise ValueError("'data' must be provided as a dense numpy array "
                             "(neurons x timepoints x trials) holding binned "
                             "spike data.")
        elif data.ndim == 2:
            data = data[:, :, None]

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
        self._initialize_storage(data.shape[0])
        self._fit_template(data)
        self._record_loss(data)

    def assert_fitted(self):
        check_is_fitted(self, ('x_knots', 'y_knots', 'template'))

    def _initialize_storage(self, n_trials):
        """
        Initializes arrays to hold loss per trial.
        """
        self._losses = np.zeros(n_trials)
        self._penalties = np.zeros(n_trials)
        self._new_losses = np.empty(n_trials)
        self._new_penalties = np.empty(n_trials)
        self.loss_hist = []
        self.penalty_hist = []
        self.objective_hist = []

    def _record_loss(self, data):
        # Compute and record reconstruction loss.
        self._eval_loss(self.x_knots, self.y_knots, self.template, data, self._losses)
        self.loss_hist.append(self._losses.mean())

        # Compute and record warping penalties.
        warp_penalties(self.x_knots, self.y_knots, self._penalties)
        self.penalty_hist.append(self._penalties.mean() * self.warp_reg_scale)

        # Record total objective history.
        self.objective_hist.append(self.loss_hist[-1] + self.penalty_hist[-1])


@numba.jit(nopython=True)
def sparsewarp(X, Y, trials, xtst, out):
    """
    Implement warping function at discrete test points, e.g. for
    spike time data.

    Parameters
    ----------
    X : ndarray
        x coordinates of knots for each trial (shape: n_trials x n_knots)
    Y : ndarray
        y coordinates of knots for each trial (shape: n_trials x n_knots)
    trials : ndarray
        trial index for each coordinate (shape: n_events)
    xtst : ndarray
        queried x coordinate for each trial (shape: n_events)

    Note
    ----
    X and Y are assumed to be sorted along axis=1

    Returns
    -------
    ytst : interpolated y value for each x in xtst (shape: n_events)
    """

    n_knots = X.shape[1]

    for i in range(len(trials)):

        x = X[trials[i]]
        y = Y[trials[i]]

        for j in range(n_knots):
            if xtst[i] <= x[j]:
                break

        if j == 0:
            slope = (y[1] - y[0]) / (x[1] - x[0])
        else:
            slope = (y[j] - y[j-1]) / (x[j] - x[j-1])

        out[i] = y[j] + slope * (xtst[i] - x[j])

    return out


@numba.jit(nopython=True)
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

        # iterate over all time bins.
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
