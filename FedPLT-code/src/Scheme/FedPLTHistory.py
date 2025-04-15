import json
import os

import numpy as np
from matplotlib import pyplot as plt

from src.SFL.SimulationHistory import SimulationHistory
from src.SFL.Tools import sprint, hhmmss



class CustomHistory(SimulationHistory):
    def __init__(self, client_counts=1, simulation_parameters=None):
        super().__init__()
        self.client_counts = client_counts

        self.simulation_parameters = simulation_parameters

        self.history = {**{i: {'training loss': [],
                               'training accuracy': [],
                               'validation loss': [],
                               'validation accuracy': [],
                               'training durations': [],
                               'selection counts': [],
                               'global-model validation loss': [],
                               'global-model validation accuracy': [],
                               'validation durations': [],
                               'testing loss': None,
                               'testing accuracy': None,
                               } for i in range(-1, self.client_counts)}
                        }

    def add_round_local_models_evolution(self, client_side_evolution):

        self.history[-1]['training loss'] += client_side_evolution['aggregated training losses']
        self.history[-1]['training accuracy'] += client_side_evolution['aggregated training accuracies']
        self.history[-1]['validation loss'] += client_side_evolution['aggregated validation losses']
        self.history[-1]['validation accuracy'] += client_side_evolution['aggregated validation accuracies']
        self.history[-1]['training durations'].append(sum(client_side_evolution['training durations']))
        self.history[-1]['selection counts'].append(client_side_evolution['selection counts'])

        sprint('info', f"Fit Results:")
        sprint('info', f"- Training Durations: {client_side_evolution['training durations']}")
        sprint('info', f"- Training Loss: {client_side_evolution['aggregated training losses']}")
        sprint('info', f"- Training Accuracy: {client_side_evolution['aggregated training accuracies']}")
        sprint('info', f"- Validation Loss: {client_side_evolution['aggregated validation losses']}")
        sprint('info', f"- Validation Accuracy: {client_side_evolution['aggregated validation accuracies']}")

        train_losses = client_side_evolution['training losses']
        train_accuracies = client_side_evolution['training accuracies']
        val_losses = client_side_evolution['validation losses']
        val_accuracies = client_side_evolution['validation accuracies']
        training_durations = client_side_evolution['training durations']

        for i in range(self.client_counts):
            self.history[i]['training loss'] += train_losses[i]
            self.history[i]['training accuracy'] += train_accuracies[i]
            self.history[i]['validation loss'] += val_losses[i]
            self.history[i]['validation accuracy'] += val_accuracies[i]
            self.history[i]['training durations'].append(training_durations[i])

    def add_round_global_model_evolution(self, server_side_evolution):
        pass

        self.history[-1]['global-model validation loss'].append(
            server_side_evolution['aggregated global-model validation losses'])
        self.history[-1]['global-model validation accuracy'].append(
            server_side_evolution['aggregated global-model validation accuracies'])
        self.history[-1]['validation durations'].append(sum(server_side_evolution['validation durations']))

        sprint('info', f"Validation Results:")
        sprint('info', f"Validation Durations: {server_side_evolution['validation durations']}")
        sprint('info', f"- Validation Loss: {server_side_evolution['aggregated global-model validation losses']}")
        sprint('info',
               f"- Validation Accuracy: {server_side_evolution['aggregated global-model validation accuracies']}")

        global_val_losses = server_side_evolution['global-model validation losses']
        global_val_accuracies = server_side_evolution['global-model validation accuracies']
        validation_durations = server_side_evolution['validation durations']

        for i in range(self.client_counts):
            self.history[i]['global-model validation loss'].append(global_val_losses[i])
            self.history[i]['global-model validation accuracy'].append(global_val_accuracies[i])
            self.history[i]['validation durations'].append(validation_durations[i])

    def register_testing_results(self, server_side_evolution: dict):

        self.history[-1]['testing loss'] = server_side_evolution['aggregated testing losses']
        self.history[-1]['testing accuracy'] = server_side_evolution['aggregated testing accuracies']

        sprint('info', f"Test Results:")
        sprint('info', f"- Testing Loss: {server_side_evolution['aggregated testing losses']}")
        sprint('info', f"- Testing Accuracy: {server_side_evolution['aggregated testing accuracies']}")

        test_losses = server_side_evolution['testing losses']
        test_accuracies = server_side_evolution['testing accuracies']

        for i in range(self.client_counts):
            self.history[i]['testing loss'] = test_losses[i]
            self.history[i]['testing accuracy'] = test_accuracies[i]

    def save(self, path: str, simulation_name: str):
        os.makedirs(path, exist_ok=True)
        
        data_to_save = {
            'simulation_parameters': self.simulation_parameters,
            'history': self.history
        }
        json_data = json.dumps(data_to_save)
        with open(f"{path}/{simulation_name}.json", 'w') as json_file:
            json_file.write(json_data)

    def load(self, path: str, simulation_name: str):
        with open(f"{path}/{simulation_name}.json", 'r') as json_file:
            loaded_data = json.load(json_file)
            self.simulation_parameters = loaded_data['simulation_parameters']
            self.history = loaded_data['history']
            self.history = {int(key): value for key, value in self.history.items()}
        self.client_counts = len(self.history) - 1

    def plot_global_model_validation(self, client_index):
        global_model_val_loss = self.history[client_index]['global-model validation loss']
        global_model_val_accuracy = self.history[client_index]['global-model validation accuracy']

        colors = np.linspace(0, 1, len(global_model_val_loss))  # Generate colors from 0 (yellow) to 1 (red)

        plt.scatter(global_model_val_loss, global_model_val_accuracy, c='green', marker='>', s=20)
        plt.plot(global_model_val_loss, global_model_val_accuracy, color='green')
        plt.xlabel('Global Models Validation Loss')
        plt.ylabel('Global Models Validation Accuracy')
        plt.title(f'Global Models Validation Loss vs. Accuracy (Client {client_index})')

        plt.show()

    def plot_durations(self):
        for i in range(2):  # Iterate over training and validation durations
            fig, ax = plt.subplots(figsize=(12, 5))  # Create a new figure for each table

            durations = []
            for client in range(self.client_counts):
                durations.append([duration for duration in (
                    self.history[client]['training durations'] if i == 0 else self.history[client][
                        'validation durations'])])

            # Add total values
            for row in durations:
                row.append(sum(row))
            durations.append([sum(column) for column in zip(*durations)])

            # Change the format
            durations = [[hhmmss(element) for element in row] for row in durations]

            row_labels = [f"Round {r}" for r in
                          range(len(durations[0]) - 1)]  # Row labels based on the length of durations
            row_labels.append("Client Duration")
            col_labels = [f"Client {c}" for c in range(self.client_counts)]
            col_labels.append("Round Duration")

            durations = [list(row) for row in zip(*durations)]

            table = ax.table(cellText=durations, rowLabels=row_labels, colLabels=col_labels, loc='center', cellLoc='center')

            # Adjust table layout
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            table.scale(0.85, 0.85)  # Adjust table scale if needed

            ax.axis('off')  # Hide axis
            ax.set_title('Training Durations' if i == 0 else 'Validation Durations')

            plt.show()

    def plot(self, target: int, save_plot: bool = False, save_path: str = './', plot_name: str = 'plot'):

        def elongate(old_list, ratio=1):
            new_list = []
            for item in old_list:
                for i in range(ratio):
                    new_list.append(item)
            return new_list

        ratio_ = int(len(self.history[target]['validation loss']) / len(
            self.history[target]['global-model validation loss']))

        node_name = 'the server' if target < 0 else f'client {target}'

        # Plot training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history[target]['training loss'], label='Training Loss')
        plt.plot(self.history[target]['validation loss'], label='Validation Loss')
        plt.plot(elongate(self.history[target]['global-model validation loss'], ratio_),
                 label='Global Models Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss of {}'.format(node_name))
        plt.legend()

        # Set the y-axis limits to start at 0 and end at 1
        plt.ylim(0)
        plt.xlim(0, len(self.history[target]['validation loss']))

        # Calculate the validation accuracy of the best model
        tradeoff = [a / (a + a * l + 1) for a, l in zip(self.history[target]['global-model validation accuracy'],
                                                        self.history[target]['global-model validation loss'])]

        arg_max_trade_off = len(tradeoff) - np.argmax(tradeoff[::-1]) - 1

        max_global_model_val_accuracy = self.history[target]['global-model validation accuracy'][arg_max_trade_off]
        arg_max_global_model_val_accuracy = arg_max_trade_off * ratio_

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history[target]['training accuracy'], label='Training Accuracy')
        plt.plot(self.history[target]['validation accuracy'], label='Validation Accuracy')
        plt.plot(elongate(self.history[target]['global-model validation accuracy'], ratio_),
                 label='Global Models Validation Accuracy')
        plt.scatter(arg_max_global_model_val_accuracy, max_global_model_val_accuracy, color='green',
                    marker='o', zorder=5)  # Place a marker on the maximum value
        plt.axhline(max_global_model_val_accuracy, color='green', linestyle='--', xmin=0,
                    xmax=arg_max_global_model_val_accuracy / 10)    # ICI
        plt.text(0, max_global_model_val_accuracy, f' {max_global_model_val_accuracy:.4f}', color='green', ha='left',
                 va='bottom')
        plt.axvline(arg_max_global_model_val_accuracy, color='green', linestyle='--', ymin=0,
                    ymax=max_global_model_val_accuracy)
        plt.text(arg_max_global_model_val_accuracy, 0, f' {arg_max_global_model_val_accuracy}', color='green', ha='left',
                 va='bottom')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy of {}'.format(node_name))
        plt.legend()

        # Set the y-axis limits to start at 0 and end at 1
        plt.ylim(0, 1)
        plt.xlim(0, len(self.history[target]['validation loss']))

        plt.tight_layout()

        # Save plot to specified directory if save_path is provided
        if save_plot:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'{plot_name}.png'))

        plt.show()

