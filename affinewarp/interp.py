from numba import jit, f8, int32, void


@jit(void(f8[:], f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:], f8[:], f8[:, :, :]), nopython=True)
def bcast_interp(xtst, X, Y, warps, template, new_loss, last_loss, data):

    T = len(xtst)
    N = len(X[0])
    n_neurons = data.shape[2]

    for i in range(len(X)):

        # do interpolation
        y0 = Y[i, 0]
        x0 = X[i, 0]
        slope = (Y[i, 1] - Y[i, 0]) / (X[i, 1] - X[i, 0])

        m = 0
        n = 1

        new_loss[i] = 0
        thres = last_loss[i]**2

        while (m < T) and (new_loss[i] < thres):

            # update interpolation point
            while (n < N-1) and (m/(T-1) > X[i, n]):
                y0 = Y[i, n]
                x0 = X[i, n]
                slope = (Y[i, n+1] - y0) / (X[i, n+1] - x0)
                n += 1

            # do interpolation and move on to next element in xtst
            z = y0 + slope*(xtst[m] - x0)

            # clip warp interpolation between zero and one
            if z < 0:
                warps[i, m] = 0.0

                # evaluate loss at first index
                for neu in range(n_neurons):
                    new_loss[i] += (template[0, neu] - data[i, m, neu]) ** 2

            elif z > 1:
                warps[i, m] = 1.0

                # evaluate loss at last index
                for neu in range(n_neurons):
                    new_loss[i] += (template[-1, neu] - data[i, m, neu]) ** 2

            else:
                warps[i, m] = z
                _i = z * (T-1)
                rem = _i % 1
                idx = int(_i)

                # evaluate loss at interpolant
                for neu in range(n_neurons):
                    new_loss[i] += (
                        (1 - rem) * template[idx, neu] +
                        rem * template[idx + 1, neu] -
                        data[i, m, neu]
                    ) ** 2

            # move to next timepoint
            m += 1

        new_loss[i] = new_loss[i] ** 0.5
