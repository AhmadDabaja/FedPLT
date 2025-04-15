import torch
import torch.nn as nn
from tqdm import tqdm
from prettytable import PrettyTable
from typing import List
import numpy as np
import random


class SegmentedLinear(nn.Module):
    def __init__(self, input_size, segment_size, number_of_segments, output=False, use_bn=False):
        super(SegmentedLinear, self).__init__()

        self.output = output
        self.fc = nn.ModuleList([nn.Linear(input_size, segment_size) for _ in range(number_of_segments)])

    def forward(self, x):

        # Use torch.jit.fork to run each layer in parallel
        futures = [torch.jit.fork(fc_segment, x) for fc_segment in self.fc]
        # Wait for all futures to complete and gather Results set A
        z = [torch.jit.wait(future) for future in futures]
        x = torch.cat(z, 1)
        if not self.output:
            x = nn.ReLU()(x)
        return x

    def freeze(self, deactivation_splits):
        for fc_segment, deactivate in zip(self.fc, deactivation_splits):
            if int(deactivate):
                for param in fc_segment.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        for fc_segment in self.fc:
            for param in fc_segment.parameters():
                param.requires_grad = True

class SegmentedConv(nn.Module):
    def __init__(self, in_channels, segment_size, number_of_segments, kernel_size, stride=1, padding=0):
        super(SegmentedConv, self).__init__()

        self.conv_segments = nn.ModuleList([
            nn.Conv2d(in_channels, segment_size, kernel_size, stride, padding)
            for _ in range(number_of_segments)
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        # Apply each segment of the convolution in parallel
        futures = [torch.jit.fork(conv, x) for conv in self.conv_segments]
        results = [torch.jit.wait(future) for future in futures]
        x = torch.cat(results, dim=1)

        x = self.activation(x)
        return x

    def freeze(self, deactivation_splits):
        for conv_segment, deactivate in zip(self.conv_segments, deactivation_splits):
            if int(deactivate):
                for param in conv_segment.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        for conv_segment in self.conv_segments:
            for param in conv_segment.parameters():
                param.requires_grad = True

class FedPLTConvFashionModel(nn.Module):

    one_random_batch = False
    layer_sizes = [1, 16, 32, 64, 128]

    def __init__(self, splits):
        super().__init__()
        self.splits = splits

        size1 = self.__class__.layer_sizes[0]

        total_splits_1 = len(splits[0])
        size2_ = self.__class__.layer_sizes[1] // total_splits_1
        size2 = size2_ * total_splits_1

        total_splits_2 = len(splits[1])
        size3_ = self.__class__.layer_sizes[2] // total_splits_2
        size3 = size3_ * total_splits_2

        total_splits_3 = len(splits[2])
        size4_ = self.__class__.layer_sizes[3] // total_splits_3
        size4 = size4_ * total_splits_3

        total_splits_4 = len(splits[3])
        size5_ = self.__class__.layer_sizes[4] // total_splits_4
        size5 = size5_ * total_splits_4

        self.layers = nn.Sequential(
            # First SegmentedConv layer + pooling
            SegmentedConv(in_channels=size1, segment_size=size2_, number_of_segments=total_splits_1, kernel_size=3,
                          stride=1, padding=1),  # Output: 28x28xsize2
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 14x14xsize2

            # Second SegmentedConv layer + pooling
            SegmentedConv(in_channels=size2, segment_size=size3_, number_of_segments=total_splits_2, kernel_size=3,
                          stride=1, padding=1),  # Output: 14x14xsize3
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 7x7xsize3

            # Third SegmentedConv layer + pooling
            SegmentedConv(in_channels=size3, segment_size=size4_, number_of_segments=total_splits_3, kernel_size=3,
                          stride=1, padding=1),  # Output: 7x7xsize4
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 3x3xsize4

            # Fourth SegmentedConv layer + pooling
            SegmentedConv(in_channels=size4, segment_size=size5_, number_of_segments=total_splits_4, kernel_size=3,
                          stride=1, padding=1),  # Output: 3x3xsize5
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 1x1xsize5
        )

        # Classification layer
        self.classifier = nn.Linear(size5, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

    def display_layers(self):
        # Create a table to display layer information
        table = PrettyTable()
        table.field_names = ["Layer Name", "Layer Type", "Output Shape", "Number of Parameters"]

        input_shape = (1, 1, 28, 28)  # Assuming MNIST input shape
        dummy_input = torch.zeros(*input_shape)

        for name, layer in self.layers.named_children():
            try:
                output = layer(dummy_input)
                dummy_input = output  # Update input for the next layer
                num_params = sum(p.numel() for p in layer.parameters())
                table.add_row([name, layer.__class__.__name__, list(output.shape), num_params])
            except Exception as e:
                table.add_row([name, layer.__class__.__name__, "N/A", "Error: " + str(e)])

        # Add the classifier layer
        num_params = sum(p.numel() for p in self.classifier.parameters())
        table.add_row(["Classifier", self.classifier.__class__.__name__, [dummy_input.size(0), 10], num_params])

        print(table)
        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")

    def freeze(self):
        for layer, split in zip(self.layers, self.splits):
            if isinstance(layer, SegmentedConv):
                layer.freeze(split)

    def unfreeze(self):
        for layer in self.layers:
            if isinstance(layer, SegmentedConv):
                layer.unfreeze()

    def train_epoch(self, train_loader, criterion, optimizer, device):
        self.train()  # Sets the module in training mode

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0  # Initialize total number of samples seen

        if FedPLTConvFashionModel.one_random_batch:
            # Select a random batch from train_loader
            random_batch = random.choice(list(train_loader))
            inputs, labels = random_batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            running_corrects += torch.sum(predicted == labels.data)
            total_samples += batch_size

        else:
            for inputs, labels in tqdm(train_loader, desc="Training", unit="step"):
                batch_size = inputs.size(0)  # Get the current batch size
                total_samples += batch_size  # Update total samples

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_size  # Scale loss by current batch size

                _, predicted = outputs.max(1)
                running_corrects += torch.sum(predicted == labels.data)

        epoch_accuracy = running_corrects.item() / total_samples  # Use total_samples
        epoch_loss = running_loss / total_samples  # Use total_samples

        return epoch_loss, epoch_accuracy

    def test_epoch(self, test_loader, criterion, device):
        self.eval()  # Sets the module in evaluation mode

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
                batch_size = inputs.size(0)
                total_samples += batch_size

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * batch_size

                _, predicted = outputs.max(1)
                running_corrects += torch.sum(predicted == labels.data)

        epoch_accuracy = running_corrects.item() / total_samples
        epoch_loss = running_loss / total_samples

        return epoch_loss, epoch_accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        # Create and return copies of the parameters as numpy arrays.
        model_parameters = [param.detach().cpu().numpy().copy() for param in self.parameters()]
        return model_parameters

    def set_model_parameters(self, parameters: List[np.ndarray]):
        # For each parameter, copy the input numpy array before converting it to a tensor.
        for param, array in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(array.copy()).to(param.device)

class FedPLTDenseFashionModel(nn.Module):

    one_random_batch = False
    layer_sizes = [1 * 28 * 28, 512, 265, 128, 10]

    def __init__(self, splits):
        super().__init__()

        self.splits = splits

        size1 = FedPLTDenseFashionModel.layer_sizes[0]

        total_splits_1 = len(self.splits[0])
        size2_ = FedPLTDenseFashionModel.layer_sizes[1] // total_splits_1
        size2 = size2_ * total_splits_1

        total_splits_2 = len(self.splits[1])
        size3_ = FedPLTDenseFashionModel.layer_sizes[2] // total_splits_2
        size3 = size3_ * total_splits_2

        total_splits_3 = len(self.splits[2])
        size4_ = FedPLTDenseFashionModel.layer_sizes[3] // total_splits_3
        size4 = size4_ * total_splits_3

        total_splits_4 = len(self.splits[3])
        size5_ = FedPLTDenseFashionModel.layer_sizes[4] // total_splits_4

        self.fc1 = SegmentedLinear(size1, size2_, total_splits_1)
        self.fc2 = SegmentedLinear(size2, size3_, total_splits_2)
        self.fc3 = SegmentedLinear(size3, size4_, total_splits_3)
        self.fc4 = SegmentedLinear(size4, size5_, total_splits_4, False)

    def forward(self, x):

        x = x.view(-1, 1 * 28 * 28)  # 4 * 4 * 256 = 4096

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    def display_layers(self):
        # Create a table
        table = PrettyTable()
        table.field_names = ["Name", "Description", "Number of Parameters"]

        # Print all layers with their parameters and additional properties for Conv layers
        for name, layer in self.named_children():
            # filtering trainable layers
            number_of_parameters = sum(parameter.numel() for parameter in layer.parameters())
            if number_of_parameters != 0:
                # Add layer to the table
                table.add_row([name, layer, number_of_parameters])

        # Print the table
        print(table)
        print('total number of parameters: {}'.format(sum(parameter.numel() for parameter in self.parameters())))

        # Training function

    def freeze(self):

        self.fc1.freeze(self.splits[0])
        self.fc2.freeze(self.splits[1])
        self.fc3.freeze(self.splits[2])
        self.fc4.freeze(self.splits[3])

    def unfreeze(self):

        self.fc1.unfreeze()
        self.fc2.unfreeze()
        self.fc3.unfreeze()
        self.fc4.unfreeze()

    def train_epoch(self, train_loader, criterion, optimizer, device):
        self.train()  # Sets the module in training mode

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0  # Initialize total number of samples seen

        if FedPLTDenseFashionModel.one_random_batch:
            # Select a random batch from train_loader
            random_batch = random.choice(list(train_loader))
            inputs, labels = random_batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            running_corrects += torch.sum(predicted == labels.data)
            total_samples += batch_size

        else:
            for inputs, labels in tqdm(train_loader, desc="Training", unit="step"):
                batch_size = inputs.size(0)  # Get the current batch size
                total_samples += batch_size  # Update total samples

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_size  # Scale loss by current batch size

                _, predicted = outputs.max(1)
                running_corrects += torch.sum(predicted == labels.data)

        epoch_accuracy = running_corrects.item() / total_samples  # Use total_samples
        epoch_loss = running_loss / total_samples  # Use total_samples

        return epoch_loss, epoch_accuracy

    # Testing function
    def test_epoch(self, test_loader, criterion, device):
        self.eval()  # Sets the module in evaluation mode

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0  # Initialize total number of samples seen

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
                batch_size = inputs.size(0)  # Get the current batch size
                total_samples += batch_size  # Update total samples

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * batch_size  # Scale loss by current batch size

                _, predicted = outputs.max(1)
                running_corrects += torch.sum(predicted == labels.data)

        epoch_accuracy = running_corrects.item() / total_samples  # Use total_samples
        epoch_loss = running_loss / total_samples  # Use total_samples

        return epoch_loss, epoch_accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        # Create and return copies of the parameters as numpy arrays.
        model_parameters = [param.detach().cpu().numpy().copy() for param in self.parameters()]
        return model_parameters

    def set_model_parameters(self, parameters: List[np.ndarray]):
        # For each parameter, copy the input numpy array before converting it to a tensor.
        for param, array in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(array.copy()).to(param.device)

class FedPLTDenseCifar10Model(nn.Module):

    one_random_batch = False
    layer_sizes = [3 * 32 * 32, 512, 265, 128, 10]

    def __init__(self, splits):
        super().__init__()

        self.splits = splits

        size1 = self.__class__.layer_sizes[0]

        total_splits_1 = len(self.splits[0])
        size2_ = self.__class__.layer_sizes[1] // total_splits_1
        size2 = size2_ * total_splits_1

        total_splits_2 = len(self.splits[1])
        size3_ = self.__class__.layer_sizes[2] // total_splits_2
        size3 = size3_ * total_splits_2

        total_splits_3 = len(self.splits[2])
        size4_ = self.__class__.layer_sizes[3] // total_splits_3
        size4 = size4_ * total_splits_3

        total_splits_4 = len(self.splits[3])
        size5_ = self.__class__.layer_sizes[4] // total_splits_4

        self.fc1 = SegmentedLinear(size1, size2_, total_splits_1)
        self.fc2 = SegmentedLinear(size2, size3_, total_splits_2)
        self.fc3 = SegmentedLinear(size3, size4_, total_splits_3)
        self.fc4 = SegmentedLinear(size4, size5_, total_splits_4, False)

    def forward(self, x):

        x = x.view(-1, 3 * 32 * 32)  # 4 * 4 * 256 = 4096

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    def display_layers(self):
        # Create a table
        table = PrettyTable()
        table.field_names = ["Name", "Description", "Number of Parameters"]

        # Print all layers with their parameters and additional properties for Conv layers
        for name, layer in self.named_children():
            # filtering trainable layers
            number_of_parameters = sum(parameter.numel() for parameter in layer.parameters())
            if number_of_parameters != 0:
                # Add layer to the table
                table.add_row([name, layer, number_of_parameters])

        # Print the table
        print(table)
        print('total number of parameters: {}'.format(sum(parameter.numel() for parameter in self.parameters())))

        # Training function

    def freeze(self):

        self.fc1.freeze(self.splits[0])
        self.fc2.freeze(self.splits[1])
        self.fc3.freeze(self.splits[2])
        self.fc4.freeze(self.splits[3])

    def unfreeze(self):

        self.fc1.unfreeze()
        self.fc2.unfreeze()
        self.fc3.unfreeze()
        self.fc4.unfreeze()

    def train_epoch(self, train_loader, criterion, optimizer, device):
        self.train()  # Sets the module in training mode

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0  # Initialize total number of samples seen

        if FedPLTDenseCifar10Model.one_random_batch:
            # Select a random batch from train_loader
            random_batch = random.choice(list(train_loader))
            inputs, labels = random_batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            running_corrects += torch.sum(predicted == labels.data)
            total_samples += batch_size

        else:
            for inputs, labels in tqdm(train_loader, desc="Training", unit="step"):
                batch_size = inputs.size(0)  # Get the current batch size
                total_samples += batch_size  # Update total samples

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_size  # Scale loss by current batch size

                _, predicted = outputs.max(1)
                running_corrects += torch.sum(predicted == labels.data)

        epoch_accuracy = running_corrects.item() / total_samples  # Use total_samples
        epoch_loss = running_loss / total_samples  # Use total_samples

        return epoch_loss, epoch_accuracy

    # Testing function
    def test_epoch(self, test_loader, criterion, device):
        self.eval()  # Sets the module in evaluation mode

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0  # Initialize total number of samples seen

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
                batch_size = inputs.size(0)  # Get the current batch size
                total_samples += batch_size  # Update total samples

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * batch_size  # Scale loss by current batch size

                _, predicted = outputs.max(1)
                running_corrects += torch.sum(predicted == labels.data)

        epoch_accuracy = running_corrects.item() / total_samples  # Use total_samples
        epoch_loss = running_loss / total_samples  # Use total_samples

        return epoch_loss, epoch_accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        # Create and return copies of the parameters as numpy arrays.
        model_parameters = [param.detach().cpu().numpy().copy() for param in self.parameters()]
        return model_parameters

    def set_model_parameters(self, parameters: List[np.ndarray]):
        # For each parameter, copy the input numpy array before converting it to a tensor.
        for param, array in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(array.copy()).to(param.device)
