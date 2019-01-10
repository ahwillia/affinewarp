from .shiftwarp import ShiftWarping
from scipy.spatial.distance import pdist, squareform
import numpy as np
from tqdm import trange


class AgglomerativeWarping:

    def __init__(self, **model_args):
        self._kwargs = model_args

    def fit(self, data):

        # Data dimensions (trials, timebins, units)
        n_trials, n_time, n_units = data.shape
        clusters = [[n] for n in range(n_units)]

        model = ShiftWarping(**self._kwargs)

        shifts = np.empty((n_units, n_trials))

        for n in trange(n_units, desc="Initializing singleton warp clusters."):
            model.fit(data[:, :, (n,)], verbose=False)
            shifts[n] = model.fractional_shifts

        print("Agglomerating...")
        for itr in range(n_units - 1):

            # Compute pairwise distances
            dists = squareform(pdist(shifts))
            dists[np.triu_indices_from(dists)] = np.inf
            dists[np.isnan(dists)] = np.inf

            # Find indices for data partitions to join (i > j).
            min_idx = np.argmin(dists.ravel())
            i, j = np.unravel_index(min_idx, dists.shape)

            print('Iteration {}/{}, merging {}, {}'.format(
                itr, n_units - 1, clusters[i], clusters[j]))

            # Merge clusters.
            clusters[j] += clusters[i]
            clusters[i] = None

            # Refit warps on merged data.
            model.fit(data[:, :, clusters[j]], verbose=False)
            shifts[j] = model.fractional_shifts
            shifts[i] = np.nan

        # Check that we've merged all clusters
        assert sum(0 if c is None else 1 for c in clusters) == 1
