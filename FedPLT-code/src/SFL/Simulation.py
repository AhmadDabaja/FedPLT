from src.SFL.Tools import sprint
from src.SFL._ClientManager import _ClientManager
from src.SFL._Server import _Server


class Simulation:
    def __init__(self, rounds_count, server_proxy, client_proxies, history=None):
        self.client_proxies = client_proxies
        self.server_proxy = server_proxy
        self.history = history

        self.clients_count = len(client_proxies)
        self.rounds_count = rounds_count

        self.server = _Server(self.server_proxy, self.clients_count)

        self.client_manager = _ClientManager()
        for cid, client_proxy, in zip(range(self.clients_count), self.client_proxies):
            self.client_manager.create_client(cid, client_proxy)
            client_pipe = self.client_manager.get_pipe(cid)
            self.server.register_client(cid, client_pipe)

    def start_simulation(self):

        sprint('info', "----------------- Initialisation -----------------")
        self.server.clients_initialisation()

        for cid in range(self.clients_count):
            self.client_manager.get_client(cid).initialization()

        for round_ in range(self.rounds_count):
            sprint('info', "---------------- Starting Round {} ----------------".format(round_ + 1))

            # The server sends the model and the training/validation configuration
            self.server.primary_upload()

            # The clients receive the model and the training/validation configuration
            # The clients train and validate the models
            # clients send trained model
            # clients send metrics and loses
            for cid in range(self.clients_count):
                self.client_manager.get_client(cid).train_validate_local_model()

            # server aggregate trained model
            client_side_evolution = self.server.primary_aggregate()
            self.history.add_round_local_models_evolution(client_side_evolution)

            # server send aggregated model
            self.server.secondary_upload()

            # client receives aggregated model
            # client validated aggregated model
            # client send losses and metrics
            for cid in range(self.clients_count):
                self.client_manager.get_client(cid).validate_global_model()

            # server aggregate trained model
            server_side_evolution = self.server.secondary_aggregate()
            self.history.add_round_global_model_evolution(server_side_evolution)

        ############################

        self.server.final_upload()

        for cid in range(self.clients_count):
            self.client_manager.get_client(cid).test_global_model()

        test_results = self.server.final_aggregate()
        self.history.register_testing_results(test_results)

    def get_history(self):
        return self.history
