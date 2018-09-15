import numpy as np
import matplotlib.pyplot as plt
# from affinewarp.piecewisewarp import warp_with_quadloss



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


# def compute_affine_loss_grid(model, data, trials=None, nshift=101, nscale=101, shift_range=(-0.5, 0.5), log_scale_range=(-2., 2.)):
#     """Compute model loss on a grid of shifts and scales.

#     Args:
#         model: PiecewiseWarping model
#         data: [n_trial, n_time, n_neuron] numpy array
#         trials: optional, default None, list of trials to evaluate
#         nshift: number of shifts
#         nscale: number of scales
#         shift_range: tuple, lower and upper bound for shfits
#         log_scale_range: tuple, log10 of lower and upper bound for scales

#     Returns:
#         shifts: numpy array, shifts where loss was evaluated
#         scales: numpy array, scales where loss was evaluated
#         losses: numpy array, shape [ntrial, nshift, nscale] with model loss for each
#           trial, shift, and scale.
#     """
#     shifts = np.linspace(*shift_range, nshift)
#     scales = np.logspace(*log_scale_range, nscale)

#     shift_grid, scale_grid = np.meshgrid(shifts, scales)
#     if trials is None:
#         trials = np.arange(data.shape[0])
#     ntrial = len(trials)
#     losses = np.zeros((ntrial, nshift * nscale))
#     x_knots, y_knots = model.x_knots[trials].copy(), model.y_knots[trials].copy()
#     for i, (shift, scale) in enumerate(zip(shift_grid.ravel(), scale_grid.ravel())):
#         y_knots[:, 0] = shift
#         y_knots[:, 1] = shift + scale
#         warp_with_quadloss(x_knots, y_knots, model.template, losses[:, i], losses[:, i], data[trials], early_stop=False)
#     losses = losses.reshape(ntrial, nshift, nscale)
#     # TODO(ahwillia): checkout NaNs in loss, likely from 0/0 in warp normalization
#     losses[np.isnan(losses)] = np.nanmax(losses)
#     return shifts, scales, losses
                             

# def visualize_affine_loss_grid(model, data, n_trials=9, **kwargs):
#     """Visualize losses for each trial, sweeping shifts and scales.

#     Args:
#         model: PiecewiseWarping model
#         data: [n_trial, n_time, n_neuron] numpy array
#         n_trials: optional, number of trials to plot
#         trials: optional, list of trials to plot
#         **kwargs: see: extra kwargs passed to compute_affine_loss_grid

#     Returns:
#         fig: figure handle
#         axs: list of axis handles 
#     """
#     trials = kwargs.get('trials', np.arange(n_trials))
#     n_trials = len(trials)
#     kwargs['trials'] = trials
#     shifts, scales, losses = compute_affine_loss_grid(model, data, **kwargs)

#     n_rows = int(np.sqrt(n_trials))
#     n_cols = int(np.ceil(n_trials / float(n_rows)))
#     fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    
#     axs = []
#     for i in range(n_trials):
#         ax = plt.subplot(n_rows, n_cols, i+1)
#         axs.append(ax)
#         plt.pcolor(shifts, np.log(scales), losses[i])
#         shift = model.y_knots[trials[i], 0]
#         scale = model.y_knots[trials[i], 1] - shift
#         ax = plt.axis()
#         plt.scatter(shift, np.log(scale), c='w', marker='*', s=50)
#         plt.axis(ax)
#         plt.title('Trial %d'%i)
#         if i == n_cols * (n_rows - 1):
#             plt.xlabel('shift')
#             plt.ylabel('log10(scale)')
#         else:
#             plt.gca().set_xticklabels([])
#             plt.gca().set_yticklabels([])

#     fig.tight_layout()
#     return fig, axs
