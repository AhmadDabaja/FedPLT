import random

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

class DataSampler:
    def __init__(self, dataset):
        # choose between CIFAR10 and SVHN datasets
        train_dataset, test_dataset = self._data_chooser(dataset)

        # Splitting into training, validation, and test sets
        val_dataset, train_dataset = self._train_val_split(train_dataset)

        # Split datasets into classes
        self.train_class_datasets = self._split_classes(train_dataset)
        self.val_class_datasets = self._split_classes(val_dataset)
        self.test_class_datasets = self._split_classes(test_dataset)

    def __call__(self, batch_size=64, sampling_ratios=None):
        # Sample data according to specified ratios
        train_dataset = self._sample_classes(self.train_class_datasets, sampling_ratios)
        val_dataset = self._sample_classes(self.val_class_datasets, sampling_ratios)
        test_dataset = self._sample_classes(self.test_class_datasets, sampling_ratios)

        # If batch_size == -1, use the entire dataset in one batch
        if batch_size == -1:
            train_batch_size = len(train_dataset)
            val_batch_size = len(val_dataset)
            test_batch_size = len(test_dataset)
        else:
            train_batch_size = batch_size
            val_batch_size = batch_size
            test_batch_size = batch_size

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

        return train_loader, valid_loader, test_loader

    def _data_chooser(self, name):
        # Check if the name is provided and is a string
        if not isinstance(name, str):
            raise ValueError("Name must be a string")

        # Define transformations based on dataset type
        if name.upper() in ['CIFAR10', 'SVHN']:
            # Transform for color images (3- with consecutive sequencing channels)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data for RGB images
            ])
        elif name.upper() in ['MNIST', 'FASHIONMNIST']:
            # Transform for grayscale images (1 channel)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Normalize the data for grayscale images
            ])
        else:
            raise ValueError("This dataset is not recognized")

        # Load the datasets based on the name provided
        if name.upper() == 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif name.upper() == 'SVHN':
            train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        elif name.upper() == 'MNIST':
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif name.upper() == 'FASHIONMNIST':
            train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                              transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                                             transform=transform)
        else:
            raise ValueError("This dataset is not recognized")

        return train_dataset, test_dataset

    def _split_classes(self, dataset):
        """
        Split dataset into 10 classes.
        """
        # Initialize a list to store datasets for each class
        classes_indices = [list() for _ in range(10)]

        # Iterate through the dataset and collect indices for each class
        for index, (_, target) in enumerate(dataset):
            classes_indices[target].append(index)

        # Convert the list of lists to CIFAR10 datasets
        class_datasets = [Subset(dataset, class_indices) for class_indices in classes_indices]

        return class_datasets

    def _sample_classes(self, class_datasets, sampling_ratios):
        """
        Take a random sample of each class according to its ratio and form a new dataset.
        """
        sampled_datasets = []
        for class_dataset, ratio in zip(class_datasets, sampling_ratios):
            num_samples = int(len(class_dataset) * ratio)
            sampled_indices = torch.randperm(len(class_dataset))[:num_samples]
            sampled_datasets.append(Subset(class_dataset, sampled_indices))
        # Concatenate the sampled datasets
        sampled_dataset = ConcatDataset(sampled_datasets)
        return sampled_dataset

    # GENERALIZE THE FUNCTION
    def _train_val_split(self, dataset):
        # Initialize dictionaries to store selected data
        val_classes = {class_idx: [] for class_idx in range(10)}
        train_classes = {class_idx: [] for class_idx in range(10)}

        # Iterate through the dataset and randomly select data
        for data, target in dataset:
            if len(val_classes[target]) < 1000:
                val_classes[target].append((data, target))
            else:
                train_classes[target].append((data, target))

        # Create datasets by combining selected data
        val_dataset = [(data, target) for class_data in val_classes.values() for data, target in class_data]
        train_dataset = [(data, target) for class_data in train_classes.values() for data, target in class_data]

        # Shuffle datasets
        random.shuffle(val_dataset)
        random.shuffle(train_dataset)

        return val_dataset, train_dataset
