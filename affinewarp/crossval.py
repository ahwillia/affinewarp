import numpy as np
from affinewarp import AffineWarping
from tqdm import trange


def kfold(N, n_splits):
    """Iterator for Kfold cross-validation
    """
    rng = np.random.permutation(N)

    stride = N / n_splits
    i = 0

    while i < n_splits:
        j = int(i * stride)
        k = int((i+1) * stride)
        test = rng[j:k]
        train = np.array(list(set(rng) - set(test)))
        test.sort()
        train.sort()
        yield train, test
        i += 1


def crossval_neurons(data, nbins, **model_params):

    # model object
    model = AffineWarping(**model_params)

    # bin data
    # binned = bin_count_data(data, nbins)

    # convert to spike time lists
    # trials, spikes = spiketimes_per_neuron(data)

    # data dimensions
    n_trials, _, n_neurons = binned.shape

    # transformed spike times
    tfm_spikes = []
    results = []

    # hold out each feature, and compute its transforms
    for n in trange(n_neurons):

        # define training set
        trainset = list(set(range(n_neurons)) - {n})

        # fit model and save parameters
        model.fit(binned[:, :, trainset], verbose=False)
        results.append(model.dump_params())

        # warp test set
        tfm_spikes.append(model.transform_events(trials[n], spikes[n]))

    return trials, spikes, tfm_spikes, results


def hyperparam_search(data, n_models=10, frac_test_trials=.25,
                      frac_test_neurons=.25, fit_iter=10):

    # dict holding results of search
    results = {
        'q1': [],
        'q2': [],
        'l2_smoothness': [],
        'nbins': [],
        'n_knots': [],
        'train_err': [],
        'test_err': [],
        'train_learning_curves': [],
        'test_learning_curves': []
    }

    nk = int()

    for m in trange(n_models):

        # sample model parameters
        params = {
            'q1': np.random.uniform(0, .5),
            'q2': np.random.uniform(0, .5),
            'l2_smoothness': np.random.uniform(0, 1000),
            'nbins': np.random.randint(100, 300),
            'n_knots': np.random.randint(0, 10)
        }

        # bin data (remove nbins temporarily)
        binned = bin_count_data(data, params.pop('nbins'))
        n_trials, n_neurons = binned.shape[0], binned.shape[2]

        # randomly sample test neurons and trials
        tst_k = np.random.choice(np.arange(n_trials), replace=False,
                                 size=int(n_trials*frac_test_trials))
        tst_n = np.random.choice(np.arange(n_neurons), replace=False,
                                 size=int(n_neurons*frac_test_neurons))

        # use remaining neurons and trials for training
        tr_k = np.array(list(set(range(n_trials)) - set(tst_k)))
        tr_n = np.array(list(set(range(n_neurons)) - set(tst_n)))
        tr_k.sort()
        tr_n.sort()

        # create model instance
        model = AffineWarping(**params)
        model.initialize_fit(binned)
        params['nbins'] = binned.shape[1]  # add nbins back

        # save training and testing error over training
        train_err, test_err = [], []

        # fit model
        for itr in range(fit_iter):
            model.fit_template(trials=tr_k)
            model.fit_warps(neurons=tr_n)

            # squared residuals
            res = (model.predict() - binned)**2

            # record training and test_error
            train_err.append(np.mean(np.concatenate(
                (res[tr_k, :, :].ravel(), res[tst_k][:, tr_n].ravel())
            )) ** .5)
            test_err.append(np.mean(res[tst_k][:, tst_n]) ** .5)

        # save model parameters
        for key in params:
            results[key].append(params[key])

        # save full learning curves for train and test set
        results['train_learning_curves'].append(train_err)
        results['test_learning_curves'].append(test_err)

        # save final training and test error across folds
        results['train_err'].append(train_err[-1])
        results['test_err'].append(test_err[-1])

    # wrap learning curves in numpy arrays
    for k in 'train_learning_curves', 'test_learning_curves':
        results[k] = np.asarray(results[k])

    return results
