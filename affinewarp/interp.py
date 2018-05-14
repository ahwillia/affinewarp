from numba import jit
import numpy as np


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
