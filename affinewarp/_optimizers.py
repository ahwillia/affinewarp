import numba
import numpy as np
import scipy as sci
import scipy.optimize


class OptimizerFactory:
    """
    Caches optimizers (fit_template, fit_warps) for different losses.
    """
    def __init__(self):
        self.cache = {}

    def __call__(self, loss):
        """Returns functions to fit template and warps given a loss function.

        Parameters
        ----------
        loss : str
            Loss function, either ('quadratic' or 'poisson'). Not
            case-sensitive.

        Returns
        -------
        template_optimizer : function
            Function that optimizes and returns warping template. Takes (data,
            template, x_knots, y_knots, smoothness_reg_scale, l2_reg_scale) as
            arguments.
        warp_optimizer : function
            Function that optimizes and returns warping template. Takes (data,
            template, x_knots, y_knots, smoothness_reg_scale, l2_reg_scale) as
            arguments.
        evaluate_loss : function
            Function that evaluates the reconstruction loss
        """
        # Create optimizer functions if necessary.
        loss = loss.lower()
        if loss not in self.cache:
            temp_opt = _construct_template_optimizer(loss)
            warp_opt, eval_loss = _construct_warp_optimizer(loss)
            self.cache[loss] = (temp_opt, warp_opt, eval_loss)

        # Return optimizers: (template_optimizer, warp_optimizer)
        return self.cache[loss]


def _construct_template_optimizer(loss):
    if loss == 'quadratic':
        # ------------------------------------------------- #
        # --- Template Update Rule Under Quadratic Loss --- #
        # ------------------------------------------------- #
        def f(x_knots, y_knots, template, data, smoothness_reg_scale, l2_reg_scale):
            K = data.shape[0]
            T = data.shape[1]

            # Initialize WtW with regularization term
            WtW = _diff_gramian(T, smoothness_reg_scale * K, l2_reg_scale * K)

            # Compute gramians.
            WtX = np.zeros((T, data.shape[-1]))
            _fast_template_grams(WtW[-2:], WtX, data, x_knots, y_knots)

            # Solve WtW * template = WtX
            return sci.linalg.solveh_banded(WtW, WtX)

    elif loss == 'poisson':
        # ----------------------------------------------- #
        # --- Template Update Rule Under Poisson Loss --- #
        # ----------------------------------------------- #
        def f(x_knots, y_knots, template, data, smoothness_reg_scale, l2_reg_scale):

            # Initialize template. Otherwise, warm-start from last result.
            if template is None:
                template = np.zeros(data.shape[1:])

            # Create objective.
            obj = PoissonObjective(data, smoothness_reg_scale, l2_reg_scale,
                                   x_knots=x_knots, y_knots=y_knots)

            opt = scipy.optimize.minimize(obj, template.ravel(),
                                          jac=True, method='L-BFGS-B',
                                          options=dict(maxiter=20))

            # # Fit using Newton's method
            # opt = scipy.optimize.minimize(obj, template.ravel(),
            #                               jac=True, hessp=obj.hessp,
            #                               method='newton-cg',
            #                               options=dict(maxiter=10))
            # print(opt.message)

            return (opt.x).reshape(template.shape)

    return f


def _construct_warp_optimizer(loss):

    # ------------------------------- #
    # --- Determine Loss Function --- #
    # ------------------------------- #
    if loss == 'quadratic':
        _loss = _quad_loss
        _interp_loss = _interp_quad_loss
    elif loss == 'poisson':
        _loss = _poiss_loss
        _interp_loss = _interp_poiss_loss
    else:
        raise ValueError("Expected 'loss' to either be 'quadratic' or "
                         "'poisson'; but saw {}.".format(loss))

    # ------------------------------------------------------------ #
    # --- Function to compute reconstruction loss on one trial --- #
    # ------------------------------------------------------------ #
    @numba.jit(nopython=True)
    def reconstruction_loss(X, Y, template, data):
        loss = 0.0

        # initialize line segement for interpolation
        slope = (Y[1] - Y[0]) / (X[1] - X[0])
        x0 = X[0]
        y0 = Y[0]

        # 'n' counts knots in piecewise affine warping function.
        n = 1

        # iterate over time bins
        for t in range(len(data)):

            # fraction of trial complete
            x = t / (len(data) - 1)

            # update interpolation point
            while (n < len(X)-1) and (x > X[n]):
                y0 = Y[n]
                x0 = X[n]
                slope = (Y[n+1] - y0) / (X[n+1] - x0)
                n += 1

            # compute index in warped time
            z = y0 + slope * (x - x0)

            # clip warp interpolation between zero and one
            if z <= 0:
                loss += _loss(template[0], data[t])

            elif z >= 1:
                loss += _loss(template[-1], data[t])

            # do linear interpolation
            else:
                j = z * (len(data) - 1)
                rem = j % 1
                loss += _interp_loss(rem, template[int(j)], template[int(j)+1], data[t])

        return loss

    # ------------------------------------------------------ #
    # --- Create function to fit warps on a single trial --- #
    # ------------------------------------------------------ #
    @numba.jit(nopython=True)
    def fit_one_warp(x_knots, y_knots, template, data, warp_reg_scale,
                     iterations, n_restarts, min_temp, max_temp,
                     initial_loss, initial_penalty,
                     curr_x, curr_y, next_x, next_y, is_shift_only):

        # Problem dimensions.
        n_knots = len(x_knots)

        # Store the best objective value seen so far.
        best_loss, best_penalty = initial_loss, initial_penalty
        best_obj = best_loss + best_penalty

        # Loss for identity warp.
        identity_loss = _loss(data.ravel(), template.ravel()) / data.size

        # Initialize to current warping knots
        for i in range(n_knots):
            curr_x[i] = x_knots[i]
            curr_y[i] = y_knots[i]

        # Do multiple random searches with refined proposal distribution.
        for start in range(n_restarts + 1):

            # First iteration starts from last set of knots. All subsequent
            # restarts start from identity warp.
            if start > 0:
                curr_loss = identity_loss
                curr_penalty = 0.0
                curr_obj = curr_loss
                for i, f in enumerate(np.linspace(0, 1, n_knots)):
                    curr_x[i] = f
                    curr_y[i] = f
            else:
                curr_loss = best_loss
                curr_penalty = best_penalty
                curr_obj = best_obj

            # Random search with exponentially decaying temperature.
            for logtemp in np.linspace(max_temp, min_temp, iterations):
                temperature = 10 ** logtemp

                # Perturb x_knots and y_knots
                for i in range(n_knots):
                    next_x[i] = curr_x[i] + temperature * np.random.randn()
                    next_y[i] = curr_y[i] + temperature * np.random.randn()

                # Sort x_knots and y_knots inplace (enforce a monotically
                # increasing warping function).
                next_x.sort()
                next_y.sort()

                # Normalize x_knots to start at zero and end at one.
                next_x = next_x - next_x[0]
                next_x = next_x / next_x[-1]

                # If desired, constrain to unit slope.
                if is_shift_only:
                    _m = (next_y[0] + next_y[1]) / 2
                    next_y[0] = _m - .5
                    next_y[1] = _m + .5

                # Compute objective on new knots.
                next_penalty = \
                    warp_penalty_one_trial(next_x, next_y) * warp_reg_scale
                next_loss = \
                    reconstruction_loss(next_x, next_y, template, data) / data.size
                next_obj = next_loss + next_penalty

                # Accept or reject next warping knots within this restart.
                if next_obj < curr_obj:
                    curr_loss = next_loss
                    curr_penalty = next_penalty
                    curr_obj = next_obj
                    for i in range(n_knots):
                        curr_x[i] = next_x[i]
                        curr_y[i] = next_y[i]

                # Save current warping knots if they're the best we've seen.
                if curr_obj < best_obj:
                    best_loss = curr_loss
                    best_penalty = curr_penalty
                    best_obj = curr_obj
                    for i in range(n_knots):
                        x_knots[i] = curr_x[i]
                        y_knots[i] = curr_y[i]

        return best_loss, best_penalty

    # ------------------------------------------------------------------ #
    # --- Create function to fit warps in parallel across all trials --- #
    # ------------------------------------------------------------------ #
    @numba.jit(nopython=True, parallel=True)
    def fit_all_warps(x_knots, y_knots, template, data, warp_reg_scale,
                      losses, penalties, iterations, n_restarts,
                      min_temp, max_temp, storage, is_shift_only):

        for k in numba.prange(x_knots.shape[0]):
            new_loss, new_pen = fit_one_warp(
                x_knots[k], y_knots[k],  # initial guess
                template, data[k],  # warping template and target
                warp_reg_scale, iterations,  # params for random search
                n_restarts, min_temp, max_temp,  # more params
                losses[k], penalties[k],
                storage[k, 0], storage[k, 1],
                storage[k, 2], storage[k, 3],
                is_shift_only
            )
            losses[k] = new_loss
            penalties[k] = new_pen

    # ---------------------------------------------------------------------- #
    # --- Create function to evaluate loss in parallel across all trials --- #
    # ---------------------------------------------------------------------- #
    @numba.jit(nopython=True, parallel=True)
    def full_loss(x_knots, y_knots, template, data, storage):
        K, T, N = data.shape
        for k in numba.prange(K):
            storage[k] = reconstruction_loss(
                x_knots[k], y_knots[k], template, data[k]) / (T * N)

    return fit_all_warps, full_loss


@numba.jit(nopython=True)
def _quad_loss(pred, targ):
    result = 0.0
    for i in range(pred.size):
        result += (pred[i] - targ[i])**2
    return result


@numba.jit(nopython=True)
def _interp_quad_loss(a, y1, y2, targ):
    result = 0.0
    b = 1 - a
    for i in range(y1.size):
        result += (b*y1[i] + a*y2[i] - targ[i])**2
    return result


@numba.jit(nopython=True)
def _poiss_loss(pred, targ):
    result = 0.0
    for i in range(pred.size):
        result += np.exp(pred[i]) - pred[i] * targ[i]
    return result


@numba.jit(nopython=True)
def _interp_poiss_loss(a, y1, y2, targ):
    result = 0.0
    b = 1 - a
    for i in range(y1.size):
        pred = (b * y1[i]) + (a * y2[i])
        result += np.exp(pred) - pred * targ[i]
    return result


@numba.jit(nopython=True)
def warp_penalty_one_trial(X, Y):
    """
    Computes penalty on warping functions for one trial.

    Parameters
    ----------
    X : ndarray, shape: (n_knots + 2,)
        x coordinates of warping function knots.
    Y : ndarray, shape: (n_knots + 2,)
        y coordinates of warping function knots.

    Returns
    -------
    penalty : float
        Penalty on warping function (not scaled).
    """

    penalty = 0.0

    for j in range(1, len(X)):

        # right point of line segment
        x0 = X[j - 1]
        y0 = Y[j - 1] - x0  # subtract off identity warp.
        x1 = X[j]
        y1 = Y[j] - x1  # subtract off identity warp.

        # If y0 and y1 have opposite signs, penalty is the area of two
        # right triangles. The height of the right triangles is y0 and
        # y1. The base of the right triangles is given by the x-intercept.
        if ((y0 < 0) and (y1 > 0)) or ((y0 > 0) and (y1 < 0)):
            v = y1 / (y1 - y0)
            penalty += 0.5 * (x1 - x0) * ((1 - v) * abs(y0) + v * abs(y1))

        # Otherwise, either one of y0 or y1 is zero, or they are both
        # positive or both negative. The penalty is the area of a trapezoid,
        # which has height x1-x0 and bases y0 and y1.
        else:
            penalty += 0.5 * abs(y0 + y1) * (x1 - x0)

    return penalty


@numba.jit(nopython=True, parallel=True)
def warp_penalties(X, Y, storage):
    for k in numba.prange(len(X)):
        storage[k] = warp_penalty_one_trial(X[k], Y[k])
    return storage


@numba.jit(nopython=True)
def warp_to_sparse_matrix(X, Y, rows, cols, vals):

    # initialize line segement for interpolation
    slope = (Y[1] - Y[0]) / (X[1] - X[0])
    x0 = X[0]
    y0 = Y[0]

    # 'n' counts knots in piecewise affine warping function.
    n = 1

    # iterate over time bins
    T = len(rows)
    for t in range(T):

        # fraction of trial complete
        x = t / (T - 1)

        # update interpolation point
        while (n < len(X)-1) and (x > X[n]):
            y0 = Y[n]
            x0 = X[n]
            slope = (Y[n+1] - y0) / (X[n+1] - x0)
            n += 1

        # compute index in warped time
        z = y0 + slope * (x - x0)

        # clip warp interpolation between zero and one
        if z <= 0:
            cols[t, 0] = t
            cols[t, 1] = t
            rows[t, 0] = 0
            rows[t, 1] = 1
            vals[t, 0] = 1.0
            vals[t, 1] = 0.0

        elif z >= 1:
            cols[t, 0] = t
            cols[t, 1] = t
            rows[t, 0] = T - 2
            rows[t, 1] = T - 1
            vals[t, 0] = 0.0
            vals[t, 1] = 1.0

        # do linear interpolation
        else:
            zf = z * (T - 1)
            cols[t, 0] = t
            cols[t, 1] = t
            rows[t, 0] = int(zf)
            rows[t, 1] = int(zf) + 1
            vals[t, 0] = 1 - (zf % 1)
            vals[t, 1] = zf % 1


class PoissonObjective:

    def __init__(self, data, smoothness_scale, l2_scale,
                 x_knots=None, y_knots=None, shifts=None):
        # Store dataset
        self.data = data.astype(np.float64)

        # Create sparse matrices representing warping functions.
        K, T, N = data.shape

        # Represent warping functions as sparse matrices.
        self.W = []

        if shifts is not None:
            # ShiftWarping models.
            rows = np.arange(T)
            vals = np.ones(T)
            for s in shifts:
                cols = np.clip(rows + s, 0, T - 1)
                Wk = scipy.sparse.csr_matrix(
                    (vals.ravel(), (rows.ravel(), cols.ravel())), shape=(T, T)
                )
                self.W.append(Wk)

        else:
            # PiecewiseWarping models.
            for x, y in zip(x_knots, y_knots):
                rows = np.empty((T, 2)).astype(int)
                cols = np.empty((T, 2)).astype(int)
                vals = np.empty((T, 2))
                warp_to_sparse_matrix(x, y, rows, cols, vals)
                Wk = scipy.sparse.csr_matrix(
                    (vals.ravel(), (rows.ravel(), cols.ravel())), shape=(T, T)
                )
                self.W.append(Wk)

        # Allocate n_timesteps x n_units matrix holding gradient and a storage
        # matrix of the same shape for intermediate calculations.
        self.grad = np.empty((T, N))
        self.hess_out = np.empty_like(self.grad)

        # Create sparse matrices for smoothing operations.
        diags = [np.ones(T-2), np.full(T-2, -2), np.ones(T-2)]
        D = scipy.sparse.diags(diags, [0, 1, 2], shape=(T-2, T))
        self.DtD = scipy.sparse.dia_matrix(D.T.dot(D))
        self.D = D

        # Store strength of smoothness and L2 regularization strengths.
        self.smoothness_scale = smoothness_scale * K
        self.l2_scale = l2_scale * K

    def __call__(self, x):
        """Computes objective and caches gradient and hessian."""

        # Reshape optimization parameters to template (T x N matrix).
        X = x.reshape(self.data.shape[1:])

        # Zero-out gradient from previous iterations
        obj = 0.0
        self.grad.fill(0.0)

        # Compute Poisson loss and gradient.
        #   - Y, count data for trial k, (time x units).
        #   - w, warping matrix for trial k (time x time).
        #   - wX, warped template, log rates, (time x units)
        #   - exp_wX, exponentiated template, rates, (time x units)
        for Y, w in zip(self.data, self.W):
            wX = w.dot(X)
            exp_wX = np.exp(wX)
            obj += (exp_wX.sum() - np.dot(Y.ravel(), wX.ravel()))
            self.grad += w.T.dot(exp_wX - Y)

        # Add smoothness penalty
        exp_X = np.exp(X)
        DtD_expX = self.DtD.dot(exp_X)
        d = self.D.dot(exp_X)
        obj += .5 * self.smoothness_scale * np.dot(d.ravel(), d.ravel())
        self.grad += self.smoothness_scale * DtD_expX * exp_X

        # Add L2 penalty
        exp_X2 = exp_X ** 2
        obj += .5 * self.l2_scale * exp_X2.sum()
        self.grad += self.l2_scale * exp_X2

        # Add contributions of smoothing and l2 regularization to gradient
        return obj, self.grad.ravel()

    def hessp(self, x, z):
        """ Hessian vector product."""
        self.hess_out.fill(0.0)

        K, T, N = self.data.shape
        X = x.reshape(T, N)
        Z = z.reshape(T, N)

        for Wk in self.W:
            WX = Wk.dot(X)
            WZ = Wk.dot(Z)
            exp_WX = np.exp(WX)
            self.hess_out += Wk.T.dot(exp_WX * WZ)

        Z_exp_X = Z * np.exp(X)
        self.hess_out += Z_exp_X * self.DtD.dot(X) + X * self.DtD.dot(X)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # TODO: add contribution from l2 #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

        return self.hess_out.ravel()


def nowarp_template(X, smoothness_scale, l2_scale, loss='quadratic'):
    """
    Solves for the optimal template assuming no warping.
    """
    K, T, N = X.shape
    if loss == 'quadratic':
        A = _diff_gramian(T, smoothness_scale * K, l2_scale * K)
        A[-1] += K
        B = np.sum(X, axis=0)
        return sci.linalg.solveh_banded(A, B)
    else:
        raise NotImplementedError


def _diff_gramian(T, smoothness_scale, l2_scale):
    """Constructs regularization term for smoothness."""
    DtD = np.ones((3, T))

    DtD[-1] = 6.0
    DtD[-1, 1] = 5.0
    DtD[-1, 0] = 1.0
    DtD[-1, -1] = 1.0
    DtD[-1, -2] = 5.0

    DtD[-2] = -4.0
    DtD[-2, 1] = -2.0
    DtD[-2, -1] = -2.0

    DtD *= smoothness_scale
    DtD[-1] += l2_scale

    return DtD


@numba.jit(nopython=True)
def _fast_template_grams(WtW, WtX, data, X, Y):
    """Computes Gram matrices for template update under quadratic loss."""

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
