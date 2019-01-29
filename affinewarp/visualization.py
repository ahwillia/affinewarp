import numpy as np
import matplotlib.pyplot as plt


def rasters(data, subplots=(5, 6), fig=None, axes=None, figsize=(9*1.5, 5*1.5),
            max_spikes=7000, style='black', **scatter_kw):
    """
    Plots a series of spike raster plots.

    Parameters
    ----------
    data : SpikeData instance
        Multi-trial spike data.
    subplots : tuple
        2-element tuple specifying number of rows and columns of subplots.
    fig : Maplotlib Figure or None
        Figure used for plotting.
    axes : ndarray of Axes objects or None
        Axes used for plotting.
    figsize : tuple
        Dimensions of figure size.
    max_spikes : int
        Maximum number of spikes to plot on a raster. Spikes are randomly
        subsampled above this limit.
    style : string
        Either ('black' or 'white') specifying background color.
    **scatter_kw
        Additional keyword args are passed to matplotlib.pyplot.scatter

    Returns
    -------
    fig : matplotlib.Figure instance
    axes : ndarray of matplotlib.Axes objects
    """

    trials, times, neurons = data.trials, data.spiketimes, data.neurons

    background = 'k' if style == 'black' else 'w'
    foreground = 'w' if style == 'black' else 'k'

    scatter_kw.setdefault('s', 1)
    scatter_kw.setdefault('lw', 0)

    # handle coloring of rasters
    if 'c' not in scatter_kw:
        scatter_kw.setdefault('color', foreground)
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
        ax.set_title('neuron {}'.format(n), color=foreground)
        ax.set_facecolor(background)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([data.tmin, data.tmax])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if fig is not None:
        fig.tight_layout()
        fig.patch.set_facecolor(background)

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
