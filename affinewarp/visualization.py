import numpy as np
import matplotlib.pyplot as plt


def rasters(trials, times, neurons, subplots=(5, 6), figsize=(9*1.5, 5*1.5),
            max_spikes=7000, **scatter_kw):

    scatter_kw.setdefault('s', 1)
    scatter_kw.setdefault('color', 'w')
    scatter_kw.setdefault('lw', 0)

    fig, axes = plt.subplots(*subplots, figsize=figsize)

    for ax, n in zip(axes.ravel(), np.unique(neurons)):
        x, y = times[neurons == n], trials[neurons == n]
        if len(x) > max_spikes:
            idx = np.random.choice(np.arange(len(x)), size=max_spikes,
                                   replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, **scatter_kw)
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
