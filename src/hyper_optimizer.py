from den_ae import main_ae
import random

SEED = 20
random.seed(SEED)


def optimize_hypers(generation_size=10, epochs=50, standard_deviation=0.1):
    assert generation_size > 0
    assert epochs > 0

    params_bounds = {
        "learning_rate": (1e-10, 1, float),
        "momentum": (0, 0.99, float),
        "lr_drop": (0, 1, float),
        "epochs_drop": (0, 20, int),
        "max_epochs": (1, 100, int),
        "l1_coeff": (1e-20, 1e-7, float),
        "l2_coeff": (1e-20, 1e-7, float),
        "zero_threshold": (0, 1e-5, float),

        "batch_size" : (1, 500, int),
        "weight_decay": (0, 1, float),
        "loss_threshold": (0, 1, float),
        "expand_by_k": (0, 50, int),

        "split_train_new_hypers": {
            "learning_rate": (1e-10, 1, float),
            "momentum": (0, 0.99, float),
            "lr_drop": (0, 1, float),
            "epochs_drop": (0, 20, int),
            "max_epochs": (1, 100, int),
            "l1_coeff": (1e-20, 1e-7, float),
            "l2_coeff": (1e-20, 1e-7, float),
            "zero_threshold": (0, 1e-5, float),
        },

        "de_train_new_hypers" : {
            "learning_rate": (1e-10, 1, float),
            "momentum": (0, 0.99, float),
            "lr_drop": (0, 1, float),
            "epochs_drop": (0, 20, int),
            "max_epochs": (1, 100, int),
            "l1_coeff": (1e-20, 1e-7, float),
            "l2_coeff": (1e-20, 1e-7, float),
            "zero_threshold": (0, 1e-5, float),
        }
    }

    # Generate initial params
    workers = []
    for i in range(generation_size):
        workers.append((0, random_init(params_bounds)))

    # Train our models
    best_worker = None
    for epoch in range(epochs):

        print("Optimization Epoch: %d/%d" % (epoch+1, epochs))
        for i, worker in enumerate(workers):
            print("Running worker: %d/%d" % (i+1, len(workers)))
            # No need to train top 20% beyond the first epoch.
            if epoch > 0 and i > int(len(workers)*0.8):
                continue

            success = False
            while not success:
                try:
                    aurocs = main_ae(worker[1])
                    auroc = sum(aurocs)/len(aurocs)

                    workers[i] = (auroc, worker[1])
                    success = True

                except Exception as e:
                    workers[i] = (worker[0], explore(worker[1], params_bounds, standard_deviation))
                    success = False
                    print("Worker %d crashed. Error:" % i)
                    print e

        # Sort the workers
        workers = sorted(workers, key=lambda x: x[0])
        best_worker = workers[-1]

        # Bottom 20% get top 20% params.
        for worker in workers[int(len(workers) * 0.2):]:
            worker[1] = exploit(workers, worker)[1]

        # Bottom 80% explores
        for worker in workers[int(len(workers) * 0.8):]:
            worker[1] = explore(worker[1], params_bounds, standard_deviation)

    return best_worker


def random_init(params_bounds):
    # Safely iterate over dict or list
    iterator = None
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
            params[key] = random_init(value)

        elif isinstance(value, list):
            params[key] = random_init(value)

        elif isinstance(value, tuple):
            lower, upper, type = params_bounds[key]

            rand = random.uniform(lower, upper)
            params[key] = type(rand)

        else:
            raise NotImplementedError

    return params


def exploit(workers, worker):
    """
    workers: List of tuples (score, params). Sorted.
    """
    selected_worker = worker
    while worker is not selected_worker and not len(workers) == 1:
        top20_top_index = len(workers)-1
        top20_bottom_index = min(len(workers) - int(0.2*len(workers)), len(workers) - 1)

        random_top20 = random.randrange(top20_bottom_index, top20_top_index)

        selected_worker = workers[random_top20]

    return selected_worker


def explore(params, param_bounds, standard_deviation=0.1):
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
        if isinstance(value, dict):
            params[key] = explore(value, param_bounds[key], standard_deviation)

        elif isinstance(value, list):
            params[key] = explore(value, param_bounds[key], standard_deviation)

        elif isinstance(value, float) or isinstance(value, int):
            lower, upper, type = param_bounds[key]

            new_value = type((standard_deviation+random.choice([-1, 1]))*value)
            new_value = max(lower, new_value)
            new_value = min(upper, new_value)

            params[key] = new_value

        else:
            raise NotImplementedError

    return params


if __name__ == "__main__":
    best_worker = optimize_hypers()
    print("Best accuracy:", best_worker[0], "with Params:", best_worker[1])

