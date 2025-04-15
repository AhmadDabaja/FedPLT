from abc import ABC, abstractmethod
from typing import List


class ServerProxy(ABC):
    @abstractmethod
    def __init__(self, global_model, server_configuration, clients_configurations):
        """


        :param global_model:
        :param server_configuration:
        :param clients_configurations:
        """
        self.global_model = global_model
        self.server_configuration = server_configuration
        self.clients_configurations = clients_configurations

    @abstractmethod
    def initialize_clients(self, cids: List[int]) -> dict:
        """
        This method must send the initial configurations for the clients.

        :param cids: (List[int]) A list of the clients' IDs.
        :return: (dict) A dictionary of dictionaries where each key represents a client ID, and the value is another
                         dictionary containing the configuration to be sent for each client (eg. the model's architecture,...).
        """
        pass

    @abstractmethod
    def pre_primary_upload(self, cids: List[int]) -> dict:
        """
        This method must prepare the training requirements (untrained parameters, ...) and the training/validation configurations to be sent then to the clients.

        :param cids: (List[int]) A list of the clients' IDs.
        :return: (dict) A dictionary of dictionaries where each key represents a client ID, and the value is another
                         dictionary containing prepared training requirements (untrained parameters, ...) and the training/validation configurations.
        """
        pass

    @abstractmethod
    def primary_aggregate(self, payloads: dict) -> dict:
        """
        This method must aggregates the models, losses, and metrics.

        :param payloads: (dict) A dictionary of dictionaries where each key represents a client ID, and the value is another
                         dictionary containing the payload received from the clients (training Parameters, losses, metrics, ...).
        :return: A dictionary containing the aggregated losses and metrics (and maybe the non-aggregated)..
        """
        pass

    @abstractmethod
    def pre_secondary_upload(self, cids: List[int]) -> dict:
        """
        This method must prepare the global model (or global parameters) and the validation configurations to be sent then to the clients.

        :param cids: (List[int]) A list of the clients' IDs
        :return: (dict) A dictionary containing the global model (or global parameters) and the validation configurations
        """
        pass

    @abstractmethod
    def secondary_aggregate(self, payloads: dict) -> dict:
        """
        This method must aggregates the losses, and the metrics.

        :param payloads: (dict) A dictionary of dictionaries where each key represents a client ID, and the value is another
                         dictionary containing the payload received from the client (losses, metrics, ...).
        :return: (dict) A dictionary containing the aggregated losses and metrics (and maybe the non-aggregated).
        """
        pass

    @abstractmethod
    def pre_final_upload(self, cids: List[int]) -> dict:
        """
        This method must prepare the global model (or global parameters) and the testing configurations to be sent then to the clients.

        :param cids: (List[int]) A list of the clients' IDs
        :return: (dict) A dictionary containing the global model (or global parameters) and the testing configurations
        """
        pass

    @abstractmethod
    def final_aggregate(self, payloads: dict) -> dict:
        """
        This method must aggregates the losses, and the metrics.

        :param payloads: (dict) A dictionary of dictionaries where each key represents a client ID, and the value is another
                         dictionary containing the payload received from the client (losses, metrics, ...).
        :return: (dict) A dictionary containing the aggregated losses and metrics (and maybe the non-aggregated).
        """
        pass
