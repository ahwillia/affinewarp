import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import trange
from sklearn.decomposition import TruncatedSVD


def rasters(data, subplots=(5, 6), figsize=(9*1.5, 5*1.5),
            max_spikes=7000, **scatter_kw):

    trials, times, neurons = data.coords

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


def dimensionality_change(data, aligned_data, sigma, **scatter_kw):

    ax = plt.gca()

    scatter_kw.setdefault('s', 1)
    scatter_kw.setdefault('color', 'k')
    scatter_kw.setdefault('lw', 0)

    n_neurons = data.shape[2]

    x, y = _dim(data, sigma), _dim(aligned_data, sigma)
    ax.scatter(x, y, **scatter_kw)

    return x, y


# def _dim(data, sigma):
#     tsvd = TruncatedSVD(n_components=1)
#     r = []
#     for n in trange(data.shape[2]):
#         x = np.asarray(data[:, :, n]).astype(float)
#         xs = gaussian_filter1d(x, sigma, axis=1)
#         tsvd.fit(xs)
#         r.append(tsvd.explained_variance_ratio_)
#     return r

def _dim(data, sigma):
    tsvd = TruncatedSVD(n_components=1)
    r = []
    for n in trange(data.shape[2]):
        x = np.asarray(data[:, :, n]).astype(float)
        xs = gaussian_filter1d(x, sigma, axis=1)
        resid = xs - np.mean(xs, axis=0, keepdims=True)
        r.append(np.sqrt(np.mean(resid**2)))
    return r
