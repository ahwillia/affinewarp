import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import trange
from sklearn.decomposition import TruncatedSVD
from .spikedata import get_spike_coords, bin_spikes


def rasters(data, subplots=(5, 6), fig=None, axes=None,
            figsize=(9*1.5, 5*1.5), max_spikes=7000, **scatter_kw):

    trials, times, neurons = get_spike_coords(data)

    scatter_kw.setdefault('s', 1)
    scatter_kw.setdefault('lw', 0)

    # handle coloring of rasters
    if 'c' not in scatter_kw:
        scatter_kw.setdefault('color', 'w')
        c = None
    else:
        c = scatter_kw.pop('c')

    if axes is None:
        fig, axes = plt.subplots(*subplots, figsize=figsize)

    for n, ax in enumerate(axes.ravel()):

        # select spikes for neuron n
        idx = np.where(neurons == n)[0]

        # turn off axis if there are no spikes
        if len(idx) == 0:
            ax.axis('off')
            continue

        # subsample spikes
        elif len(idx) > max_spikes:
            idx = np.random.choice(idx, size=max_spikes, replace=False)

        # make raster plot
        if c is not None:
            ax.scatter(times[idx], trials[idx], c=c[idx], **scatter_kw)
        else:
            ax.scatter(times[idx], trials[idx], **scatter_kw)

        # format axes
        ax.set_title('neuron {}'.format(n), color='w')
        ax.set_facecolor('k')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if fig is not None:
        fig.tight_layout()
        fig.patch.set_facecolor('k')

    return fig, axes


def binned_heatmap(binned, subplots=(5, 6), figsize=(9*1.5, 5*1.5), **kwargs):

    kwargs.setdefault('aspect', 'auto')

    fig, axes = plt.subplots(*subplots, figsize=figsize)

    for ax, n in zip(axes.ravel(), range(binned.shape[-1])):
        ax.imshow(binned[:, :, n], **kwargs)
        ax.set_facecolor('k')
        ax.set_title('neuron {}'.format(n), color='w')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes.ravel()[n:]:
        ax.axis('off')

    fig.tight_layout()
    fig.patch.set_facecolor('k')

    return fig, axes


def stacked_raster_psth(raw, aligned, nplots=6, figsize=(9*1.5, 5*1.5),
                        height_ratio=.5, nbins=100, **raster_kw):

    fig, axes = plt.subplots(3, nplots, figsize=figsize,
                             gridspec_kw={
                             'height_ratios': [1, 1, height_ratio]
                             })

    rasters(raw, axes=axes[0], fig=fig, **raster_kw)
    rasters(aligned, axes=axes[1], fig=fig, **raster_kw)

    binned_raw = bin_spikes(raw, nbins)
    binned_align = bin_spikes(aligned, nbins)

    raw_m = binned_raw.mean(axis=0)
    raw_sd = binned_raw.std(axis=0) / np.sqrt(raw.shape[0])
    align_m = binned_align.mean(axis=0)
    align_sd = binned_align.std(axis=0) / np.sqrt(aligned.shape[0])
    x = np.arange(len(raw_m))

    for n, ax in enumerate(axes[-1]):
        ax.plot(x, raw_m[:, n], color='k')
        ax.plot(x, align_m[:, n], color='r')

        ax.fill_between(x,
                        raw_m[:, n] + raw_sd[:, n],
                        raw_m[:, n] - raw_sd[:, n],
                        color='k', alpha=.3, zorder=-1)

        ax.fill_between(x,
                        align_m[:, n] + align_sd[:, n],
                        align_m[:, n] - align_sd[:, n],
                        color='r', alpha=.3, zorder=-1)

    fig.tight_layout()

    return fig, axes
