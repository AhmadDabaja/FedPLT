import random


def get_model_class_by_name(model_name):
    """
    Dynamically returns the model class from src.Models given its class name as a string.

    Parameters:
    - model_name (str): Name of the model class as a string.
                        e.g., 'FedPLTDenseFashionModel', 'FedPLTDenseCifar10Model', ...

    Returns:
    - class: The model class corresponding to the provided name.

    Raises:
    - ValueError: If the class name is not found in src.Models.
    """
    from src import Models

    if hasattr(Models, model_name):
        return getattr(Models, model_name)
    else:
        raise ValueError(f"Model '{model_name}' not found in src.Models.")


def create_deactivation_function(clients_count, layer_sizes, pause_vector_list, setup_name):
    """
    Creates a callable function that returns a randomized list of deactivation vectors and
    their corresponding shrinking ratios for use in federated learning.

    Parameters:
    - clients_count (int): Number of clients to generate deactivation vectors for.
    - layer_sizes (list of int): Sizes of the model layers, e.g., [784, 128, 64, 10].
                                Used to compute total parameter count per layer.
    - pause_vector_list (list of tuples): Each tuple contains:
        - A deactivation vector (list of floats between 0 and 1) indicating the
          proportion of units to deactivate in each layer.
        - A proportion (float between 0 and 1) indicating how many clients should
          use that vector (as a fraction of `clients_count`).
    - setup_name (str): The name assigned to the returned function.

    Returns:
    - fn (callable): When called, returns a tuple:
        - deactivation_vectors (list of lists): One deactivation vector per client.
        - shrinking_ratios (list of floats): Each ratio represents the proportion of
          parameters deactivated for the corresponding client.
    """

    def fn():
        # Expand pause_vector_list into a list of deactivation vectors for clients
        deactivation_vectors = [
            vector.copy()  # Copy to avoid shared references between clients
            for vector, proportion in pause_vector_list
            for _ in range(int(clients_count * proportion))
        ]

        # If we have fewer vectors than clients (due to rounding), duplicate randomly
        all_vectors = [vector for vector, _ in pause_vector_list]
        while len(deactivation_vectors) < clients_count:
            deactivation_vectors.append(random.choice(all_vectors).copy())

        def calculate_shrinking_ratios(neuron_sizes, deact_vector):
            """
            Compute the ratio of parameters deactivated given a deactivation vector.
            """
            # Calculate parameter count per layer (weights + biases)
            layer_params = [
                neuron_sizes[i - 1] * neuron_sizes[i] + neuron_sizes[i]
                for i in range(1, len(neuron_sizes))
            ]
            total_param_count = sum(layer_params)
            shrunk_param_count = sum(d * s for d, s in zip(deact_vector, layer_params))

            return shrunk_param_count / total_param_count

        # Compute shrinking ratios for all clients
        shrinking_ratios = [
            calculate_shrinking_ratios(layer_sizes, dv)
            for dv in deactivation_vectors
        ]

        # Shuffle vectors and ratios together to randomize client assignment
        paired = list(zip(deactivation_vectors, shrinking_ratios))
        random.shuffle(paired)
        deactivation_vectors, shrinking_ratios = map(list, zip(*paired))

        return deactivation_vectors, shrinking_ratios

    fn.__name__ = setup_name  # Give the returned function a name for debugging/inspection
    return fn
