import copy
from copy import deepcopy

import numpy as np

from src.SFL.ServerProxy import ServerProxy
from src.SFL.Tools import sprint

from fractions import Fraction

# WITH DYNAMIC LEARNING RATE


def choose_deactivated_splits(activation_ratio):
    # Calculate the deactivation ratio element-wise
    deactivation_ratio = 1 - activation_ratio

    # Convert each element to Fraction and limit the denominator
    deactivation_ratio_fractions = np.vectorize(lambda x: Fraction(x).limit_denominator())(deactivation_ratio)

    # Use lambda functions to extract numerators and denominators
    clients_deactivation_count_per_layer = np.vectorize(lambda x: x.numerator)(deactivation_ratio_fractions)
    total_splits = np.vectorize(lambda x: x.denominator)(deactivation_ratio_fractions)

    # splits_count_per_layer = np.max(total_splits, axis=0)
    splits_count_per_layer = np.apply_along_axis(lambda x: np.lcm.reduce(x), axis=0, arr=total_splits)
    clients_deactivation_count_per_layer = clients_deactivation_count_per_layer * (splits_count_per_layer // total_splits)

    clients_count = clients_deactivation_count_per_layer.shape[0]
    splits_counts = clients_deactivation_count_per_layer.shape[1]

    # Determine the choices for each citizen
    clients_splits = []

    # Track the current candidate index for each seat
    current_splits_index = np.zeros(splits_counts, dtype=int)

    for client_idx in range(clients_count):
        client_choices = []
        for split_idx in range(splits_counts):
            num_choices = clients_deactivation_count_per_layer[client_idx, split_idx]
            chosen_splits = []
            while num_choices > 0:
                # Find the next available candidate index for this seat
                choosen_split_idx = current_splits_index[split_idx] % splits_count_per_layer[split_idx]
                chosen_splits.append(choosen_split_idx)
                current_splits_index[split_idx] += 1
                num_choices -= 1

            # Convert choices to a binary string representation
            binary_chosen_splits = ''.join(
                '1' if idx in chosen_splits else '0' for idx in range(splits_count_per_layer[split_idx]))
            client_choices.append(binary_chosen_splits)

        clients_splits.append(client_choices)

    return clients_splits


class Client_Selector():
    def __init__(self, clients_ids, m, clients_shrinking_ratios, original_ocs):
        # Initialize all probabilities to 1.0
        self.clients_ids = clients_ids
        self.m = m
        self.clients_shrinking_ratios = clients_shrinking_ratios
        self.selection_probabilities = {cid: 1.0 for cid in clients_ids}
        self.selected_clients = None
        self.original_ocs = original_ocs

    def update_probabilities(self, clients_gradient_norms, clients_data_counts):
        """
        Update selection probabilities using the derived algorithm.

        Parameters:
          clients_gradient_norms: list or array of gradient norms, one per client.
          clients_data_counts: list or array of data counts d_i.
        """
        # Convert inputs to numpy arrays for convenience.
        grad_norms = np.array(clients_gradient_norms, dtype=float)
        data_counts = np.array(clients_data_counts, dtype=float)
        if not self.original_ocs:
            alphas = np.array(self.clients_shrinking_ratios, dtype=float) # change to np.ones_like for exp 2
        else:
            alphas = np.ones_like(self.clients_shrinking_ratios, dtype=float)
        N = len(self.clients_ids)

        # Compute score_i = sqrt(alpha_i)*d_i*grad_norm
        scores = np.sqrt(alphas) * data_counts * grad_norms

        total_alpha = np.sum(alphas)

        # Determine set S.
        # We work with indices corresponding to the clients.
        indices = np.arange(N)
        # Order indices by increasing score.
        ordered_indices = indices[np.argsort(scores)]

        S = []  # indices in S
        cum_alpha = 0.0

        # First, add indices until cumulative alpha exceeds (total_alpha - m)
        for idx in ordered_indices:
            if cum_alpha <= total_alpha - self.m:
                S.append(idx)
                cum_alpha += alphas[idx]
            else:
                break

        # Now, from the remaining indices, try adding them one by one
        # using the inequality check.
        # The inequality is:
        #   sqrt(alpha_k)*d_k*grad_norm_k < (sum_{j in S∪{k}} sqrt(alpha_j)*d_j*grad_norm_j) / (m - total_alpha + sum_{j in S} alpha_j)
        # Here, note that m - total_alpha + sum_{j in S} alpha_j must be > 0.
        S_set = set(S)
        for idx in ordered_indices:
            if idx in S_set:
                continue  # already in S
            # Compute the candidate numerator if we add idx.
            new_S = list(S_set.union({idx}))
            sum_S = np.sum([np.sqrt(alphas[j]) * data_counts[j] * grad_norms[j] for j in new_S])
            denominator = self.m - total_alpha + np.sum([alphas[j] for j in new_S])
            # Check the inequality for candidate idx:
            if np.sqrt(alphas[idx]) * data_counts[idx] * grad_norms[idx] < sum_S / denominator:
                S_set.add(idx)
            else:
                # If the inequality fails, we do not add idx.
                # Depending on the algorithm, you may break here;
                # here we assume no further addition is allowed.
                break

        # Convert S_set to sorted list (optional).
        S = sorted(list(S_set))

        # Now compute probabilities p_i for i in S and p_i = 1 for i in S^c.
        # The formula for i in S:
        # p_i = ((m - total_alpha + sum_{j in S} alpha_j) / sqrt(alpha_i)) * (d_i * grad_norm_i / (sum_{j in S} sqrt(alpha_j)*d_j*grad_norm_j))
        sum_S_term = np.sum([np.sqrt(alphas[j]) * data_counts[j] * grad_norms[j] for j in S])
        constant_factor = self.m - total_alpha + np.sum([alphas[j] for j in S])

        new_probabilities = {}
        for i, cid in enumerate(self.clients_ids):
            if i in S:
                p = (constant_factor / np.sqrt(alphas[i])) * (data_counts[i] * grad_norms[i] / sum_S_term)
                # Ensure p is between 0 and 1.
                p = min(max(p, 0.0), 1.0)
                new_probabilities[cid] = p
            else:
                new_probabilities[cid] = 1.0

        self.selection_probabilities = new_probabilities

    def __call__(self):
        # Perform client selection based on a Bernoulli trial using the computed probabilities.
        self.selected_clients = {cid: bool(np.random.binomial(1, prob)) for cid, prob in self.selection_probabilities.items()}
        return self.selected_clients


class PLTServerProxy(ServerProxy):
    def __init__(self, global_model, server_configuration, clients_configurations):
        super().__init__(global_model, server_configuration, clients_configurations)

        deactivation_function = server_configuration['deactivation_function']
        pause_vectors, clients_shrinking_ratios = deactivation_function()
        self.aggregation_method = self.server_configuration['aggregation_method']
        client_selection_count = self.server_configuration['client_selection_count']
        original_ocs = self.server_configuration['original_ocs']

        cids = list(range(len(pause_vectors)))

        pause_vectors.append([1, 1, 1, 1])
        self.clients_splits = choose_deactivated_splits(np.array(pause_vectors))

        self.client_selector = Client_Selector(cids, client_selection_count, clients_shrinking_ratios, original_ocs)

        self.best = {'model': None, 'trade-off': 0}

    def initialize_clients(self, cids):
        payloads = {}
        for cid in cids:
            local_model = deepcopy(self.global_model)(self.clients_splits[cid])

            client_configuration = self.clients_configurations[cid]
            payloads[cid] = {'local-model': local_model,
                             'client configuration': client_configuration}

        device = self.clients_configurations[0]['device']
        self.global_model = self.global_model(self.clients_splits[-1])
        self.global_model.to(device)

        return payloads

    def pre_primary_upload(self, cids):
        payloads = {}

        client_selected = self.client_selector()

        for cid in cids:
            local_learning_parameters = self.global_model.get_model_parameters()
            payloads[cid] = {'local-model': local_learning_parameters, 'selected': client_selected[cid]}

        sprint('debug', "The server is sending the untrained local model parameters to the clients ...")

        return payloads

    def primary_aggregate(self, payloads):

        inverted_payloads = {}
        for cid, payload in payloads.items():
            for key, value in payload.items():
                if key not in inverted_payloads:
                    inverted_payloads[key] = []
                inverted_payloads[key].append(value)

        trained_gradients, gradients_norms, train_data_counts, train_losses, train_accuracies, val_losses, val_accuracies, training_durations = inverted_payloads.values()

        performance = {'training losses': train_losses,
                       'training accuracies': train_accuracies,
                       'validation losses': val_losses,
                       'validation accuracies': val_accuracies,
                       'training durations': training_durations,
                       'aggregated training losses': None,
                       'aggregated training accuracies': None,
                       'aggregated validation losses': None,
                       'aggregated validation accuracies': None,
                       'selection counts': None}


        selected_clients = list(self.client_selector.selected_clients.values())
        shrinking_ratios = self.client_selector.clients_shrinking_ratios


        # Populating the dictionary
        counts = {}

        for ratio, selected in zip(shrinking_ratios, selected_clients):
            if ratio not in counts.keys():
                counts[ratio] = {0: 0, 1: 0}
            counts[ratio][selected] += 1

        performance['selection counts'] = counts

        sprint('debug', 'The server is aggregating the weights, metrics, and losses ...')

        if self.aggregation_method in ['DSAvg', 'PartDSAvg']:
            aggregated_gradients = [np.zeros_like(param) for param in self.global_model.get_model_parameters()]
            data_counts_sums = [0.0 for _ in aggregated_gradients]  # ✅ FIX: Move this outside the per-layer loop

            for gradients, train_data_count in zip(trained_gradients, train_data_counts):
                if gradients is None:  # ✅ FIX: Handle missing clients
                    continue

                for l, gradient_tensor in enumerate(gradients):
                    if np.all(gradient_tensor == 0):
                        continue  # Skip zero gradients

                    aggregated_gradients[l] += train_data_count * gradient_tensor  # Weighted sum
                    data_counts_sums[l] += train_data_count

            # Normalize each layer separately
            for l in range(len(aggregated_gradients)):
                if data_counts_sums[l] > 0:
                    aggregated_gradients[l] /= data_counts_sums[l]

        elif self.aggregation_method == 'PartAvg':
            # Initialize lists to store the aggregated gradients
            aggregated_gradients = [np.zeros_like(param) for param in self.global_model.get_model_parameters()]
            num_clients_per_layer = [0 for _ in aggregated_gradients]  # Track valid clients per layer

            for gradients in trained_gradients:
                if gradients is None:
                    continue  # Skip clients that didn't send gradients

                for l, gradient_tensor in enumerate(gradients):
                    if np.all(gradient_tensor == 0):
                        continue  # Skip zero gradients

                    aggregated_gradients[l] += gradient_tensor  # Sum gradients
                    num_clients_per_layer[l] += 1  # Count clients that contributed to this layer

            # Normalize each layer separately
            for l in range(len(aggregated_gradients)):
                if num_clients_per_layer[l] > 0:
                    aggregated_gradients[l] /= num_clients_per_layer[l]  # Average over contributing clients

        else:
            raise ValueError('Aggregation method not implemented')

        # ✅ FIX: Ensure correct update of model parameters
        current_params = self.global_model.get_model_parameters()
        new_parameters = [param + gradient for param, gradient in zip(current_params, aggregated_gradients)]
        self.global_model.set_model_parameters(new_parameters)

        # weighting the metrics and losses
        weighted_train_losses = [[epoch_train_loss * train_data_count for epoch_train_loss in client_train_losses] for
                                 client_train_losses, train_data_count in
                                 zip(train_losses, train_data_counts)]
        weighted_train_accuracies = [
            [epoch_train_accuracy * train_data_count for epoch_train_accuracy in client_train_accuracies] for
            client_train_accuracies, train_data_count in
            zip(train_accuracies, train_data_counts)]
        weighted_val_losses = [
            [epoch_val_loss * train_data_count for epoch_val_loss in client_val_losses] for
            client_val_losses, train_data_count in
            zip(val_losses, train_data_counts)]
        weighted_val_accuracies = [
            [epoch_val_accuracy * train_data_count for epoch_val_accuracy in client_val_accuracies] for
            client_val_accuracies, train_data_count in
            zip(val_accuracies, train_data_counts)]

        # Aggregating losses and metrics
        performance['aggregated training losses'] = [sum(epoch_weighted_train_losses) / sum(train_data_counts) for
                                                     epoch_weighted_train_losses in zip(*weighted_train_losses)]
        performance['aggregated training accuracies'] = [sum(epoch_weighted_train_accuracies) / sum(train_data_counts)
                                                         for epoch_weighted_train_accuracies in
                                                         zip(*weighted_train_accuracies)]
        performance['aggregated validation losses'] = [
            sum(epoch_weighted_val_losses) / sum(train_data_counts) for epoch_weighted_val_losses in
            zip(*weighted_val_losses)]
        performance['aggregated validation accuracies'] = [
            sum(epoch_weighted_val_accuracies) / sum(train_data_counts) for epoch_weighted_val_accuracies in
            zip(*weighted_val_accuracies)]

        self.client_selector.update_probabilities(gradients_norms, train_data_counts)

        return performance


    def pre_secondary_upload(self, cids):

        payloads = {}
        for cid in cids:
            global_model_learning_parameters = self.global_model.get_model_parameters()
            payloads[cid] = {'global model parameters': global_model_learning_parameters}

        sprint('debug', "The server is sending the trained global parameters to the clients ...")

        return payloads


    def secondary_aggregate(self, payloads: dict) -> dict:

        inverted_payloads = {}
        for cid, payload in payloads.items():
            for key, value in payload.items():
                if key not in inverted_payloads:
                    inverted_payloads[key] = []
                inverted_payloads[key].append(value)

        val_data_counts, global_val_losses, global_val_accuracies, validation_durations = inverted_payloads.values()

        performance = {'global-model validation losses': global_val_losses,
                       'global-model validation accuracies': global_val_accuracies,
                       'validation durations': validation_durations,
                       'aggregated global-model validation losses': None,
                       'aggregated global-model validation accuracies': None}

        sprint('debug', 'The server is aggregating the metrics, and losses ...')

        # weighting the metrics and losses
        weighted_global_val_losses = [client_global_val_loss * val_data_count for
                                      client_global_val_loss, val_data_count in
                                      zip(global_val_losses, val_data_counts)]
        weighted_global_val_accuracies = [client_sub_val_accuracy * val_data_count for
                                          client_sub_val_accuracy, val_data_count in
                                          zip(global_val_accuracies, val_data_counts)]

        # Aggregating losses and metrics
        performance['aggregated global-model validation losses'] = sum(weighted_global_val_losses) / sum(
            val_data_counts)
        performance['aggregated global-model validation accuracies'] = sum(weighted_global_val_accuracies) / sum(
            val_data_counts)

        # updating the best model
        l = performance['aggregated global-model validation losses']
        a = performance['aggregated global-model validation accuracies']
        trade_off = a / (a + a * l + 1)

        if trade_off >= self.best['trade-off']:
            self.best['trade-off'] = trade_off
            self.best['model'] = self.global_model.get_model_parameters()

        return performance

    def pre_final_upload(self, cids):
        best_global_model_learning_parameters = self.best['model']

        payloads = {}
        for cid in cids:
            payloads[cid] = {'best global model parameters': copy.deepcopy(best_global_model_learning_parameters)}

        sprint('debug', "The server is sending the best global parameters to the clients ...")

        return payloads

    def final_aggregate(self, payloads):

        inverted_payloads = {}
        for cid, payload in payloads.items():
            for key, value in payload.items():
                if key not in inverted_payloads:
                    inverted_payloads[key] = []
                inverted_payloads[key].append(value)

        test_data_counts, testing_losses, testing_accuracies = inverted_payloads.values()

        performance = {'testing losses': testing_losses,
                       'testing accuracies': testing_accuracies,
                       'aggregated testing loss': None,
                       'aggregated testing accuracies': None}

        sprint('debug', 'The server is aggregating the metrics, and losses ...')

        # weighting the metrics and losses
        weighted_testing_losses = [client_testing_loss * test_data_count for
                                   client_testing_loss, test_data_count in
                                   zip(testing_losses, test_data_counts)]
        weighted_testing_accuracies = [client_testing_accuracy * test_data_count for
                                       client_testing_accuracy, test_data_count in
                                       zip(testing_accuracies, test_data_counts)]

        # Aggregating losses and metrics
        performance['aggregated testing losses'] = sum(weighted_testing_losses) / sum(
            test_data_counts)
        performance['aggregated testing accuracies'] = sum(weighted_testing_accuracies) / sum(
            test_data_counts)

        return performance


