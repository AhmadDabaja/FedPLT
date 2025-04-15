import copy

from torch import optim

from src.SFL.ClientProxy import ClientProxy
from src.SFL.Tools import sprint, StopWatch

import numpy as np

# WITH DYNAMIC LEARNING RATE


class PLTClientProxy(ClientProxy):
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__(train_loader, val_loader, test_loader)

        self.local_model = None

        self.learning_rate = None
        self.criterion = None
        self.epochs = None
        self.device = None

    def configure(self, payload: dict):
        self.local_model = payload['local-model']
        client_configuration = payload['client configuration']

        self.learning_rate = client_configuration['learning rate']
        self.epochs = client_configuration['epochs']
        self.criterion = client_configuration['criterion']
        self.device = client_configuration['device']

        self.local_model.to(self.device)

    def fit(self, payload):
        # the client receives untrained sub_model parameters and set them to its local model

        local_learning_parameters = payload['local-model']
        selected = payload['selected']
        stop_watch = StopWatch()  # starting a counting timer
        self.local_model.set_model_parameters(local_learning_parameters)
        self.local_model.freeze()

        # Initialize losses and metrics lists
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # setting the training optimiser
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.epochs):
            sprint('debug', 'train {}'.format(epoch))
            train_loss, train_accuracy = self.local_model.train_epoch(self.train_loader,
                                                                      self.criterion,
                                                                      optimizer,
                                                                      self.device)
            sprint('debug', 'test {}'.format(epoch))
            val_loss, val_accuracy = self.local_model.test_epoch(self.val_loader,
                                                                 self.criterion,
                                                                 self.device)

            # Append losses and accuracies
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        self.local_model.unfreeze()

        trained_weights = self.local_model.get_model_parameters()

        # find the gradient and its norm
        gradients = []
        for new_param, old_param in zip(trained_weights, local_learning_parameters):
            gradient = new_param - old_param
            gradients.append(gradient)

        gradient_norm = np.linalg.norm(np.concatenate([gradient.flatten() for gradient in gradients]))

        train_data_count = len(self.train_loader.dataset)

        # registering the training duration
        training_duration = stop_watch.stop()

        payload = {'sub-model gradients': gradients if selected else None,
                   'sub_model gradient norm': gradient_norm,
                   'training data samples count': train_data_count,
                   'training losses': train_losses,
                   'training accuracies': train_accuracies,
                   'validation losses': val_losses,
                   'validation accuracies': val_accuracies,
                   'training duration': training_duration}

        return payload

    def val(self, payload):
        # the client receives trained global model parameters and set them to its local model
        global_model_learning_parameters = payload['global model parameters']
        stop_watch = StopWatch()  # starting a counting timer
        # self.local_model.reattain_super_model()
        self.local_model.set_model_parameters(global_model_learning_parameters)
        self.local_model.to(self.device)

        # Test the model
        super_val_loss, super_val_accuracy = self.local_model.test_epoch(self.val_loader, self.criterion, self.device)

        val_data_count = len(self.val_loader.dataset)

        # registering the training duration
        validation_duration = stop_watch.stop()

        payload = {'validation data samples count': val_data_count,
                   'global-model validation losses': super_val_loss,
                   'global-model validation accuracies': super_val_accuracy,
                   'validation duration': validation_duration}

        return payload

    def test(self, payload):
        # the client receives the best global model parameters and set them to its local model
        best_global_model_learning_parameters = payload['best global model parameters']
        self.local_model.set_model_parameters(best_global_model_learning_parameters)
        self.local_model.to(self.device)

        # Test the model
        test_loss, test_accuracy = self.local_model.test_epoch(self.test_loader, self.criterion, self.device)

        test_data_count = len(self.test_loader.dataset)

        payload = {'testing data samples count': test_data_count,
                   'testing loss': test_loss,
                   'testing accuracy': test_accuracy}

        return payload
