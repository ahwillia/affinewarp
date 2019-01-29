"""Experimental code on Agglomerative Warping."""

from .shiftwarp import ShiftWarping
from scipy.spatial.distance import pdist, squareform
import numpy as np
from tqdm import tqdm, trange
import itertools
from copy import deepcopy


class AgglomerativeWarping:

    def __init__(self, **model_args):
        self._kwargs = model_args

    def fit(self, data):

        # Data dimensions (trials, timebins, units)
        n_trials, n_time, n_units = data.shape
        clusters = [[n] for n in range(n_units)]

        model = ShiftWarping(**self._kwargs)
        shifts = np.empty((n_units, n_trials))
        loss = np.empty((n_units,))

        for n in trange(n_units, desc="Initializing singleton warp clusters"):
            model.fit(data[:, :, (n,)], verbose=False)
            shifts[n] = model.fractional_shifts
            loss[n] = model.loss_hist[-1]

        self.loss_hist = [np.mean(loss)]
        self.corr_hist = []
        self.all_corrs = []

        for itr in trange(n_units - 1, desc="Agglomerating clusters"):

            # Compute pairwise distances
            dists = squareform(pdist(shifts, metric='correlation'))
            dists[np.triu_indices_from(dists)] = np.inf
            dists[np.isnan(dists)] = np.inf

            # Find indices for data partitions to join (i > j).
            min_idx = np.argmin(dists.ravel())
            i, j = np.unravel_index(min_idx, dists.shape)
            self.corr_hist.append(1.0 - dists[i, j])
            self.all_corrs.append(dists[np.isfinite(dists)].ravel())

            # Merge clusters.
            clusters[j] += clusters[i]
            clusters[i] = None

            # Refit warps on merged data.
            model.fit(data[:, :, clusters[j]], verbose=False)
            shifts[j] = model.fractional_shifts
            shifts[i] = np.nan

            loss[i] = np.nan
            loss[j] = model.loss_hist[-1] * len(clusters[j])
            self.loss_hist.append(np.nansum(loss) / n_units)

        # Check that we've merged all clusters
        assert sum(0 if c is None else 1 for c in clusters) == 1


class StagewiseWarping:

    def __init__(self, **model_args):
        self._kwargs = model_args

    def fit(self, data):

        # Data dimensions (trials, timebins, units)
        n_trials, n_time, n_units = data.shape

        # List of neuron indices for each cluster. Start with singleton clusters.
        clusters = [[n] for n in range(n_units)]

        # Compute costs associated with each singleton cluster.
        model = ShiftWarping(**self._kwargs)
        indiv_costs = np.empty(n_units)
        for n in trange(n_units, desc="Fitting singleton clusters"):
            model.fit(data[:, :, (n,)], verbose=False)
            indiv_costs[n] = model.loss_hist[-1]

        # Compute cost after merging all possible pairs of clusters.
        combs = itertools.combinations(range(n_units), 2)
        ncombs = n_units * (n_units - 1) // 2
        joint_costs = np.full((n_units, n_units), np.inf)
        for i, j in tqdm(combs, total=ncombs, desc="Fitting pairwise clusters."):
            model.fit(data[:, :, [i, j]], verbose=False)
            joint_costs[i, j] = 2 * model.loss_hist[-1]

        self.loss_hist = [np.mean(indiv_costs)]
        self.cluster_hist = [deepcopy(clusters)]

        # Merge neurons into clusters iteratively.
        for itr in trange(n_units - 1, desc="Agglomerating clusters."):

            # Find clusters i and j to merge to minimize cost.
            merge_costs = joint_costs - indiv_costs[:, None] - indiv_costs[None, :]
            min_idx = np.argmin(merge_costs.ravel())
            i, j = np.unravel_index(min_idx, merge_costs.shape)

            # Update individual costs (attribute to cluster i, delete cluster j)
            cluster_size = len(clusters[i]) + len(clusters[j])
            indiv_costs[i] = cluster_size * joint_costs[i, j]
            indiv_costs[j] = 0.0

            # Store total loss
            self.loss_hist.append(np.sum(indiv_costs) / n_units)

            # Merge cluster j into cluster i.
            clusters[i] += clusters[j]
            self.cluster_hist.append(deepcopy(clusters))

            # Delete cluster j.
            clusters[j] = None
            joint_costs[j] = np.inf
            joint_costs[:, j] = np.inf

            # Update joint costs containing cluster i.
            for k in range(n_units):
                if (k != i) and clusters[k]:
                    neurons = clusters[k] + clusters[i]
                    model.fit(data[:, :, neurons], verbose=False)
                    if k > i:
                        joint_costs[k, i] = len(neurons) * model.loss_hist[-1]
                    else:
                        joint_costs[i, k] = len(neurons) * model.loss_hist[-1]

        # Check that we've merged all clusters
        assert sum(0 if c is None else 1 for c in clusters) == 1
