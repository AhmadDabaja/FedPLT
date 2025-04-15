import numpy as np

from .DataSampler import DataSampler
from .DataDrawer import DataDrawer

def dirichlet_sampler(dataset, clients_count=10, classes_count=10, alphas=[10000], batch_size=64, uniform_data_count=False):
    if clients_count % len(alphas) != 0:
        raise ValueError("Clients count must be divisible by the length of alphas.")

    train_loaders = []
    val_loaders = []
    test_loaders = []
    group_size = clients_count // len(alphas)

    if not uniform_data_count:
        while True:
            sampling_ratios = np.concatenate([np.random.dirichlet([alpha] * group_size, classes_count) for alpha in alphas], axis=1)

            # Check if any column contains a zero
            if not np.any(np.all(np.round(sampling_ratios, 2) == 0, axis=0)):
                break
    else:
        def root_dot_averaging(vector, n):
            log_product = np.sum(np.log(vector))  # Sum of logarithms
            nth_root = np.exp(log_product / n)  # Exponentiate to get the nth root
            return nth_root

        while True:
            # Generate skewed sampling ratios using Dirichlet distribution
            sampling_ratios = np.concatenate([np.random.dirichlet([alpha] * group_size, classes_count) for alpha in alphas], axis=1)
            sampling_ratios = sampling_ratios / np.sum(sampling_ratios, 0)
            sampling_ratios = sampling_ratios / np.max(np.sum(sampling_ratios, 1))

            if root_dot_averaging(np.sum(sampling_ratios, axis=1), classes_count) > 0.99 and not np.any(np.all(np.round(sampling_ratios, 2) == 0, axis=0)):
                break

    data_sampler = DataSampler(dataset)

    for cid in range(clients_count):
        client_sampling_ratios = sampling_ratios[:, cid]
        train_loader, val_loader, test_loader = data_sampler(batch_size, client_sampling_ratios)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders


def dirichlet_drawer(dataset, clients_count=10, classes_count=10, alphas=[10000], batch_size=64, uniform_data_count=False):
    if clients_count % len(alphas) != 0:
        raise ValueError("Clients count must be divisible by the length of alphas.")

    train_loaders = []
    val_loaders = []
    test_loaders = []
    group_size = clients_count // len(alphas)

    if not uniform_data_count:
        while True:
            sampling_ratios = np.concatenate([np.random.dirichlet([alpha] * group_size, classes_count) for alpha in alphas], axis=1)

            # Check if any column contains a zero
            if not np.any(np.all(np.round(sampling_ratios, 2) == 0, axis=0)):
                break
    else:
        def root_dot_averaging(vector, n):
            log_product = np.sum(np.log(vector))  # Sum of logarithms
            nth_root = np.exp(log_product / n)  # Exponentiate to get the nth root
            return nth_root

        while True:
            # Generate skewed sampling ratios using Dirichlet distribution
            # sampling_ratios = np.concatenate([np.random.dirichlet([alpha] * group_size, classes_count) for alpha in alphas], axis=1)
            sampling_ratios = np.random.dirichlet(alphas, classes_count)
            sampling_ratios = sampling_ratios / np.sum(sampling_ratios, 0)
            sampling_ratios = sampling_ratios / np.max(np.sum(sampling_ratios, 1))

            if root_dot_averaging(np.sum(sampling_ratios, axis=1), classes_count) > 0.95 and not np.any(np.all(np.round(sampling_ratios, 2) == 0, axis=0)):
                break


    data_drawer = DataDrawer(dataset)

    for cid in range(clients_count):
        client_sampling_ratios = sampling_ratios[:, cid]
        train_loader, val_loader, test_loader = data_drawer(batch_size, client_sampling_ratios)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders
