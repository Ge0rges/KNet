"""
File contains functions to search for best parameters.
"""
import random
import numpy as np

from torch.multiprocessing import Pool, cpu_count, set_start_method
from sklearn.decomposition import PCA
from src.main_scripts.den_ae import main_ae


def optimize_hypers(generation_size=8, epochs=10, standard_deviation=0.1, use_cuda=False, data_loaders=None,
                    num_workers=0, classes_list=None, criterion=None, seed=None, error_function=None,
                    encoder_in=None, hidden_encoder=None, hidden_action=None, action_out=None, core_invariant_size=None,
                    params_bounds=None, workers_seed=None):
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
    assert data_loaders is not None
    assert classes_list is not None
    assert criterion is not None
    assert error_function is not None
    assert generation_size > 0
    assert epochs > 0
    assert params_bounds is not None
    assert workers_seed is None or len(workers_seed) <= generation_size

    if seed is not None:
        random.seed(seed)

    if use_cuda:
        set_start_method('spawn')

    # Generate initial params
    workers = []

    autoencoder_out = int(core_invariant_size) if core_invariant_size is not None else None
    if autoencoder_out is None or autoencoder_out <= 0:
        print("Doing PCA on the data...")
        autoencoder_out = []
        for dl in data_loaders:
            autoencoder_out.append(pca_dataset(data_loader=dl, threshold=0.9))
        autoencoder_out = int(max(autoencoder_out))

    print("Initializing workers...")
    workers.extend(workers_seed)
    for i in range(generation_size-len(workers_seed)):
        workers.append((0, random_init(params_bounds, autoencoder_out, encoder_in, hidden_encoder,
                                       hidden_action, action_out)))


    # Train our models
    best_worker = None
    for epoch in range(epochs):

        print("Optimization Epoch: %d/%d" % (epoch+1, epochs))

        # Multithreading!
        generation_size -= generation_size % cpu_count()

        pool = Pool(min(generation_size, cpu_count()))
        args = []
        for i in range(len(workers)):
            i_args = [i, epoch, workers[i], len(workers), error_function, use_cuda, data_loaders, num_workers,
                      classes_list, criterion, seed]
            args.append(i_args)

        workers = pool.starmap(train_worker, args)

        pool.close()
        pool.join()

        # # Linear :(
        # workers_new = []
        # for i in range(len(workers)):
        #     result = train_worker(i, epoch, workers[i], len(workers), error_function, use_cuda, data_loader, num_workers,
        #                                                  classes_list, criterion, seed)
        #     workers_new.append(result)
        # workers = workers_new

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

def train_worker(i, epoch, worker, workers_len, error_function, use_cuda, data_loader, num_workers, classes_list,
                 criterion, seed):
    """
    Trains one worker.
    """

    print("Running worker: %d/%d" % (i + 1, workers_len))
    # No need to train top 20% beyond the first epoch.
    if epoch > 0 and i > int(workers_len * 0.8):
        return worker

    try:
        save_model_name = None  # Change to save: str(i) + "_model_epoch" + str(epoch) + ".pt"
        model, perfs = main_ae(worker[1], worker[1]["split_train_new_hypers"], worker[1]["de_train_new_hypers"],
                        error_function, use_cuda, data_loader, num_workers, classes_list, criterion, save_model_name,
                        seed)
        perf = sum(perfs) / len(perfs)

        worker = (perf, worker[1], model)

        return worker

    except Exception as e:
        print("worker " + str(i) + " crashed:" + str(e))
        return (0, worker[1], None)

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

        elif key is not "sizes":
            raise NotImplementedError

    return params


def construct_network_sizes(autoencoder_out, encoder_in, hidden_encoder, hidden_action, action_out):
    sizes = {}

    def power_law(input, output, number_of_layers, layer_number):
        exp = np.log(input) - np.log(output)
        exp = np.divide(exp, np.log(number_of_layers))
        result = input/np.power(layer_number, exp)

        return result

    # AutoEncoder
    middle_layers = []

    for i in range(2, hidden_encoder+2):
        current = int(power_law(encoder_in, autoencoder_out, hidden_encoder+2, i))
        if current <= autoencoder_out or current <= 1:
            break
        middle_layers.append(current)

    sizes["encoder"] = [int(encoder_in)] + middle_layers + [int(autoencoder_out)]

    # Action
    middle_layers = []
    for i in range(2, hidden_action+2):
        current = int(power_law(autoencoder_out, action_out, hidden_action+2, i))
        if current <= autoencoder_out or current <= 1:
            break
        middle_layers.append(current)

    sizes["action"] = [int(autoencoder_out)] + middle_layers + [int(action_out)]

    return sizes


def pca_dataset(data_loader=None, threshold=0.9):
    assert data_loader is not None

    # Most of the time, the datasets are too big to run PCA on it all, so we're going to get a random subset
    # that hopefully will be representative
    train, valid, test = data_loader.get_loaders()
    train_data = []
    for i, (input, target) in enumerate(train):
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

