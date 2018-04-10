from numba import jit, f8, int32, void


@jit(void(f8[:], f8[:, :], f8[:, :], int32[:], f8[:, :], f8[:, :, :], f8[:, :]), nopython=True)
def bcast_interp(xtst, X, Y, jumps, warps, pred, template):

    T = len(template)

    for i in range(len(X)):

        # find jumps
        for j in range(0, len(jumps)):

            left = 0
            right = len(xtst) - 1
            # jumps[j] = -1

            while left <= right:
                mid = left + ((right-left)//2)
                if xtst[mid] > X[i, j+1]:
                    jumps[j] = mid
                    right = mid - 1
                else:
                    left = mid + 1

            # if jumps[j] == -1:
            #     raise AssertionError('jumps are malformed')

        # do interpolation
        y0 = Y[i, 0]
        x0 = X[i, 0]
        slope = (Y[i, 1] - Y[i, 0]) / (X[i, 1] - X[i, 0])

        m = 0
        n = 0

        while m < T:

            # update interpolating points
            if n < len(jumps) and (m == jumps[n]):
                y0 = Y[i, n]
                x0 = X[i, n]
                slope = (Y[i, n+1] - y0) / (X[i, n+1] - x0)
                n += 1

            # do interpolation and move on to next element in xtst
            else:
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
                    pred[i, m] = (1-rem)*template[int(idx)] + rem*template[int(idx)+1]

                    for neuron in range(n_neurons):
                        losses[i] += (pred[i, m, neuron] - data[i, m, neuron]) ** 2

                m += 1

        losses[i] = losses[i] ** 0.5
