from torch import save, load
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
from collections import Counter

def save_dataloaders(train_loaders, test_loaders, val_loaders, filename):
    """
    Save the datasets used by training, testing, and validation DataLoader objects to a file.

    Parameters:
        train_loaders (list): A list of training DataLoader objects.
        test_loaders (list): A list of testing DataLoader objects.
        val_loaders (list): A list of validation DataLoader objects.
        filename (str): The file path to save the datasets. The datasets will be stored in a format that can be reloaded later.

    This function extracts the dataset objects from the provided DataLoader instances
    and saves them into a single file. Only the datasets are saved, which allows for
    flexibility in recreating DataLoader instances with different configurations in the future.
    """
    data_loader_info = {
        'train': [],
        'test': [],
        'val': []
    }

    for loader in train_loaders:
        data_loader_info['train'].append(loader.dataset)

    for loader in test_loaders:
        data_loader_info['test'].append(loader.dataset)

    for loader in val_loaders:
        data_loader_info['val'].append(loader.dataset)

    save(data_loader_info, filename)
    print(f"Datasets saved to {filename}.")


def load_dataloaders(filename, batch_size=64, shuffle=True):
    """
    Load datasets and create DataLoader objects for training, testing, and validation.

    Parameters:
        filename (str): The file path from which to load the datasets. This file should contain the datasets
                        saved previously using the `save_dataloaders` function.
        batch_size (int, optional): The number of samples per batch to be loaded. Default is 64.
                                    If set to -1, the entire dataset will be loaded as one batch.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Default is True.

    Returns:
        tuple: Three lists containing the loaded training, testing, and validation DataLoader objects.
    """
    data_loader_info = load(filename)

    train_loaders = []
    test_loaders = []
    val_loaders = []

    for dataset in data_loader_info['train']:
        loader = DataLoader(
            dataset,
            batch_size=len(dataset) if batch_size == -1 else batch_size,
            shuffle=shuffle,
        )
        train_loaders.append(loader)

    for dataset in data_loader_info['test']:
        loader = DataLoader(
            dataset,
            batch_size=len(dataset) if batch_size == -1 else batch_size,
            shuffle=shuffle,
        )
        test_loaders.append(loader)

    for dataset in data_loader_info['val']:
        loader = DataLoader(
            dataset,
            batch_size=len(dataset) if batch_size == -1 else batch_size,
            shuffle=shuffle,
        )
        val_loaders.append(loader)

    print(f"DataLoaders loaded from {filename}.")
    return train_loaders, test_loaders, val_loaders


def plot_stacked_class_distribution(filename):
    """
    Load training datasets from a file and plot a stacked bar chart showing class distribution per client.

    Parameters:
        filename (str): Path to the .pt file containing datasets.

    Returns:
        None
    """
    # Load the dataset
    data_loader_info = load(filename)
    train_datasets = data_loader_info['train']  # List of training datasets
    num_clients = len(train_datasets)

    # Collect class distributions
    class_distributions = []
    all_classes = set()

    for dataset in train_datasets:
        labels = [label for _, label in dataset]  # Extract labels
        class_count = Counter(labels)
        class_distributions.append(class_count)
        all_classes.update(class_count.keys())

    all_classes = sorted(list(all_classes))  # Sort class labels

    # Prepare data for stacked bar plot
    client_indices = np.arange(num_clients)
    bottom_values = np.zeros(num_clients)  # Track bottom of stacked bars

    plt.figure(figsize=(12, 6))

    for class_label in all_classes:
        class_counts = [class_distributions[i].get(class_label, 0) for i in range(num_clients)]
        plt.bar(client_indices, class_counts, label=f'Class {class_label}', bottom=bottom_values)
        bottom_values += np.array(class_counts)  # Update bottom for stacking

    plt.xlabel("Clients")
    plt.ylabel("Number of Samples")
    plt.title("Stacked Class Distribution Across Clients")
    plt.xticks(client_indices, [f'Client {i+1}' for i in range(num_clients)])
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_sample_distribution(dataset_path: str, bin_width: int = 64):
    """
    Plot the distribution of number of samples per training DataLoader.

    Args:
        dataset_path (str): Path to the dataset file.
    """

    training_loaders, _, _ = load_dataloaders(dataset_path)

    all_sample_sizes = [len(loader.dataset) for loader in training_loaders]


    bin_count = int(max(all_sample_sizes) / bin_width)

    plt.figure(figsize=(10, 6))
    plt.hist(all_sample_sizes, bins=bin_count, edgecolor='black', density=False)
    plt.title('Distribution of Number of Samples Per DataLoader')
    plt.xlabel('Number of Samples')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()