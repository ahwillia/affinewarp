import numpy as np
from affinewarp import AffineWarping


def crossval_neurons(data, **kwargs):

    # trials, timepoints, features
    K, T, N = data.shape

    # holds transformed data
    tfm = np.empty_like(data)
    all_models = []

    # hold out each feature, and compute its transforms
    modeldicts = []
    for n in range(N):

        # fit on training data
        trainset = list(set(range(N)) - {n})
        model = AffineWarping(data[:, :, trainset])
        model.fit()

        # warp test set
        tfm[:, :, n] = model.transform(data[:, :, n])

        # save other data
        all_models.append(model)

    return tfm, all_models
