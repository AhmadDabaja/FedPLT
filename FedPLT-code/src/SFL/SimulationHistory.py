from abc import abstractmethod, ABC


class SimulationHistory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_round_local_models_evolution(self, client_side_evolution: dict):
        """
        This method must store the losses and metrics obtained from the local models for each round.

        :param client_side_evolution: (dict) A dictionary containing the losses and metrics.
        :return:
        """
        pass

    @abstractmethod
    def add_round_global_model_evolution(self, server_side_evolution: dict):
        """
        This method must store the losses and metrics obtained from the global models for each round.

        :param server_side_evolution: (dict) A dictionary containing the losses and metrics.
        :return:
        """
        pass

    @abstractmethod
    def register_testing_results(self, server_side_evolution: dict):
        """
        This method must store the loss and metrics obtained from the final global models.

        :param server_side_evolution: (dict) A dictionary containing the losses and metrics.
        :return:
        """
        pass

    @abstractmethod
    def save(self, path: str, simulation_name: str):
        """
        This method must save the history of a simulation.

        :param path: (str) The path to which the simulation history will be saved.
        :param simulation_name: (str) The name of this simulation; also the name of the file where the simulation history will be stored.
        :return:
        """
        pass

    @abstractmethod
    def load(self, path: str, simulation_name: str):
        """
        This method must load the history of a simulation.

        :param path: (str) (str) The path from which the simulation history will be loaded.
        :param simulation_name: (str) The name of this simulation; also the name of the file where the simulation history will be loaded from.
        :return:
        """
        pass
