from numba import jit, f8, int32, void


@jit(void(f8[:], f8[:, :], f8[:, :], f8[:, :], f8[:, :, :], f8[:, :], f8[:], f8[:], f8[:, :, :]), nopython=True)
def bcast_interp(xtst, X, Y, warps, pred, template, new_loss, last_loss, data):

    T = len(xtst)
    N = len(X[0])

    for i in range(len(X)):

        # do interpolation
        y0 = Y[i, 0]
        x0 = X[i, 0]
        slope = (Y[i, 1] - Y[i, 0]) / (X[i, 1] - X[i, 0])

        m = 0
        n = 1

        new_loss[i] = 0

        while (m < T) and (new_loss[i] < last_loss[i]**2):

            # update interpolation point
            while (n < N-1) and (m/(T-1) > X[i, n]):
                y0 = Y[i, n]
                x0 = X[i, n]
                slope = (Y[i, n+1] - y0) / (X[i, n+1] - x0)
                n += 1

            # do interpolation and move on to next element in xtst
            else:
                # putative interpolated value
                z = y0 + slope*(xtst[m] - x0)

                if z < 0:
                    warps[i, m] = 0.0
                    pred[i, m] = template[0]

                elif z > 1:
                    warps[i, m] = 1.0
                    pred[i, m] = template[-1]

                else:
                    warps[i, m] = z
                    idx = z * (T-1)
                    rem = idx % 1
                    pred[i, m, :] = (1-rem)*template[int(idx), :] + rem*template[int(idx)+1, :]

                    new_loss[i] += ((pred[i, m, :] - data[i, m, :]) ** 2).sum()

                m += 1

        new_loss[i] = new_loss[i] ** 0.5
