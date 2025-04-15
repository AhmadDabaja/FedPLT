from abc import abstractmethod, ABC
from typing import Dict


class ClientProxy(ABC):
    @abstractmethod
    def __init__(self, train_loader, val_loader, test_loader):
        """

        :param train_loader:
        :param val_loader:
        :param test_loader:
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    @abstractmethod
    def configure(self, payload: dict):
        """
        This method must configure this client with the configurations sent by the server

        :param payload: (dict) A dictionary containing the configuration of the client received by the server (eg. the model's architecture,...).
        """
        pass

    @abstractmethod
    def fit(self, payload: Dict) -> Dict:
        """
        This method must Fit the model sent in the payload according to the configuration provided in the payload.

        :param payload: (dict) A dictionary containing the training requirement (untrained model/training parameters,...) and the training/validation configuration.
        :return: A dictionary containing the trained model (or training Parameters) and the resulted losses and metrics.
        """
        pass

    @abstractmethod
    def val(self, payload: Dict) -> Dict:
        """
        This method must validate the global model sent in the payload according to the configuration provided in the payload.

        :param payload: (dict) A dictionary containing the trained global model (or global parameters) and the validation configuration.
        :return: A dictionary containing the resulted losses and metrics.
        """
        pass

    @abstractmethod
    def test(self, payload: Dict) -> Dict:
        """
        This method must test the final global model sent in the payload according to the configuration provided in the payload.

        :param payload: (dict) A dictionary containing the final global model (or global parameters) and the validation configuration.
        :return: A dictionary containing the resulted losses and metrics.
        """
        pass

