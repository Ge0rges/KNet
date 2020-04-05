"""
File contains functions to search for best parameters.
"""
import itertools
import random

from src.main_scripts.den_ae import main_ae
from src.utils.data_loading import EEG_bands_to_binary_loader
from torch.multiprocessing.pool import Pool
from torch.multiprocessing import cpu_count
import numpy as np
from sklearn.decomposition import PCA


def optimize_hypers(generation_size=8, epochs=10, standard_deviation=0.1, use_cuda=False, data_loader=None,
                    num_workers=0, classes_list=None, criterion=None, seed=None, error_function=None,
                    encoder_in=None, hidden_encoder=None, hidden_action=None, action_out=None, params_bounds=None):
    """
    Trains generation_size number of models for epochs number of times.
    At every epoch the bottom 20% workers copy the top 20%
    At every epoch the bottom 80% of workers explore their parameters.
    Returns the best hyperparameters found for running den_ae().

    Recommend setting generation size as a multiple of cpu_count()
    """

    assert action_out is not None
    assert hidden_action is not None
    assert hidden_encoder is not None
    assert encoder_in is not None
    assert data_loader is not None
    assert classes_list is not None
    assert criterion is not None
    assert error_function is not None
    assert generation_size > 0
    assert epochs > 0
    assert params_bounds is not None

    if seed is not None:
        random.seed(seed)

    # Generate initial params
    workers = []

    print("Doing PCA on the data...")
    autoencoder_out = pca_dataset(data_loader=data_loader, threshold=0.9)

    for i in range(generation_size):
        workers.append((0, random_init(params_bounds, autoencoder_out, encoder_in, hidden_encoder,
                                       hidden_action, action_out)))
    print("Done PCA.")

    # Train our models
    best_worker = None
    for epoch in range(epochs):

        print("Optimization Epoch: %d/%d" % (epoch+1, epochs))

        # # Multithreading!
        # pool = Pool(min(generation_size, cpu_count()))
        # args = itertools.zip(range(len(workers)), itertools.repeat(epoch), workers, itertools.repeat(len(workers)),
        #                       itertools.repeat(error_function),  itertools.repeat(use_cuda),
        #                       itertools.repeat(data_loader),  itertools.repeat(num_workers),
        #                       itertools.repeat(classes_list), itertools.repeat(criterion),  itertools.repeat(seed)
        #                     )
        # workers = pool.map(train_worker_star, args)
        #
        # pool.close()
        # pool.join()

        workers_new = []
        for i in range(len(workers)):
            result = train_worker(i, epoch, workers[i], len(workers), error_function, use_cuda, data_loader, num_workers,
                                                         classes_list, criterion, seed)
            workers_new.append(result)
        workers = workers_new

        # Sort the workers
        workers = sorted(workers, key=lambda x: x[0])
        best_worker = workers[-1]
        print("At epoch {} got best worker: {}".format(epoch, best_worker))

        # Bottom 20% get top 20% params.
        for i, worker in enumerate(workers[:int(len(workers) * 0.2)]):
            workers[i] = (worker[0], exploit(workers, worker)[1])

        # Bottom 80% explores
        for i, worker in enumerate(workers[:int(len(workers) * 0.8)]):
            workers[i] = (worker[0], explore(worker[1], params_bounds, standard_deviation))

    return best_worker


def train_worker_star(args):
    """
    Calls train_worker(*args).
    """
    return train_worker(*args)


def train_worker(i, epoch, worker, workers_len, error_function, use_cuda, data_loader, num_workers, classes_list,
                 criterion, seed):
    """
    Trains one worker.
    """

    print("Running worker: %d/%d" % (i + 1, workers_len))
    # No need to train top 20% beyond the first epoch.
    if epoch > 0 and i > int(workers_len * 0.8):
        return worker

    save_model_name = str(i) + "_model_epoch" + str(epoch) + ".pt"
    perfs = main_ae(worker[1], worker[1]["split_train_new_hypers"], worker[1]["de_train_new_hypers"],
                    error_function, use_cuda, data_loader, num_workers, classes_list, criterion, save_model_name,
                    seed)
    perf = sum(perfs) / len(perfs)

    worker = (perf, worker[1])

    return worker


def random_init(params_bounds, autoencoder_out, encoder_in, hidden_encoder, hidden_action, action_out):
    """
    Randomly initializes the parameters within their bounds.
    """
    # Safely iterate over dict or list
    if isinstance(params_bounds, dict):
        iterator = params_bounds.items()

    elif isinstance(params_bounds, list):
        iterator = enumerate(params_bounds)

    else:
        raise NotImplementedError

    # Build the params
    params = {}

    for key, value in iterator:
        if isinstance(value, dict):
            params[key] = random_init(value, autoencoder_out, encoder_in, hidden_encoder, hidden_action, action_out)

        elif isinstance(value, list):
            params[key] = random_init(value, autoencoder_out, encoder_in, hidden_encoder, hidden_action, action_out)

        elif isinstance(value, tuple):
            lower, upper, type = params_bounds[key]

            rand = random.uniform(lower, upper)
            params[key] = type(rand)

        else:
            raise NotImplementedError

    # Sizes
    params["sizes"] = construct_network_sizes(autoencoder_out, encoder_in, hidden_encoder, hidden_action, action_out)
    return params


def exploit(workers, worker):
    """
    Worker copies one of the top 20%.
    workers: List of tuples (score, params). Sorted.

    >>> workers = [(0.9, {"param0":"m"}), (0.5, {"param2":"m"}), (0.6, {"param1":"m"}), (0.2, {"param4":"m"}), (0.2, {"param3":"m"})]
    >>> exploit(workers, worker[4])
    (0.9, {"param0":"m"})
    """
    selected_worker = worker
    while worker is not selected_worker and not len(workers) == 1:
        top20_top_index = len(workers)-1
        top20_bottom_index = min(len(workers) - int(0.2*len(workers)), len(workers) - 1)

        random_top20 = random.randrange(top20_bottom_index, top20_top_index)

        selected_worker = workers[random_top20]

    return selected_worker


def explore(params, param_bounds, standard_deviation=0.1):
    """
    Params are modified to be either increased or decreased by roughly one standard deviation.
    They remain within their bounds.
    """
    # Safely iterate
    iterator = None
    if isinstance(params, dict):
        iterator = params.items()

    elif isinstance(params, list):
        iterator = enumerate(params)

    else:
        raise NotImplementedError

    # Recursive calls till base case
    for key, value in iterator:
        if isinstance(value, dict) and key is not "sizes":
            params[key] = explore(value, param_bounds[key], standard_deviation)

        elif isinstance(value, list):
            params[key] = explore(value, param_bounds[key], standard_deviation)

        elif isinstance(value, float) or isinstance(value, int):
            lower, upper, type = param_bounds[key]

            new_value = type(standard_deviation*random.choice([-1, 1])) if value == 0 else \
                            type((standard_deviation*random.choice([-1, 1])+1)*value)
            new_value = max(lower, new_value)
            new_value = min(upper, new_value)
                                                      
            params[key] = new_value

        else:
            raise NotImplementedError

    return params


def construct_network_sizes(autoencoder_out, encoder_in, hidden_encoder, hidden_action, action_out):
    sizes = {}

    # AutoEncoder
    middle_layers = []

    previous = encoder_in
    for i in range(hidden_encoder):
        if previous/2 <= autoencoder_out or previous <= 1:
            break
        middle_layers.append(int(previous/2))

    sizes["encoder"] = [int(encoder_in)] + middle_layers + [int(autoencoder_out)]

    # Action
    middle_layers = []
    previous = autoencoder_out
    for i in range(hidden_action):
        if previous / 2 <= action_out or previous <= 1:
            break
        middle_layers.append(int(previous/2))
        
    sizes["action"] = [int(autoencoder_out)] + middle_layers + [int(action_out)]

    return sizes


def pca_dataset(data_loader=None, threshold=0.9):
    assert data_loader is not None

    # Most of the time, the datasets are too big to run PCA on it all, so we're going to get a random subset
    # that hopefully will be representative
    train, valid, test = data_loader()
    train_data = []
    for i, (input, target) in enumerate(train):
        n = input.size()[0]
        indices = np.random.choice(list(range(n)), size=(int(n/5)))
        input = input.numpy()
        data = input[indices]
        train_data.extend(data)

    for i, (input, target) in enumerate(valid):
        n = input.size()[0]
        indices = np.random.choice(list(range(n)), size=(int(n/5)))
        input = input.numpy()
        data = input[indices]
        train_data.extend(data)

    for i, (input, target) in enumerate(test):
        n = input.size()[0]
        indices = np.random.choice(list(range(n)), size=(int(n/5)))
        input = input.numpy()
        data = input[indices]
        train_data.extend(data)

    train_data = np.array(train_data)
    model = PCA()
    model.fit_transform(train_data)
    var = model.explained_variance_ratio_.cumsum()
    n_comp = 0
    vars = []
    for i in var:
        vars.append(i)
        if i >= threshold:
            n_comp += 1
            break
        else:
            n_comp += 1

    return n_comp

