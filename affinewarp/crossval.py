import numpy as np
from affinewarp import AffineWarping
from .utils import spiketimes_per_neuron, bin_count_data
from tqdm import trange


def crossval_neurons(data, nbins, **model_params):

    # bin data
    binned = bin_count_data(data, nbins)

    # convert to spike time lists
    trials, spikes = spiketimes_per_neuron(data)

    # data dimensions
    n_trials, _, n_neurons = binned.shape

    # model
    model = AffineWarping(**model_params)

    # transformed spike times
    tfm_spikes = []
    model_params = []

    # hold out each feature, and compute its transforms
    for n in trange(n_neurons):

        # define training set
        trainset = list(set(range(n_neurons)) - {n})
        model.initialize_fit(binned[:, :, trainset])

        # fit model and save parameters
        model.fit(verbose=False)
        model_params.append(model.dump_params())

        # warp test set
        tfm_spikes.append(model.transform_events(trials[n], spikes[n]))

    return trials, spikes, tfm_spikes, model_params
