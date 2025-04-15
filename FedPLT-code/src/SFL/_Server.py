from src.SFL.Tools import sprint


class _Server:
    def __init__(self, server_proxy, clients_count):
        self.server_proxy = server_proxy
        self.clients_count = clients_count

        self.pipes = {}

    def register_client(self, cid, pipe):
        self.pipes[cid] = pipe

    def clients_initialisation(self):
        # for each client sent its model and his configuration
        cids = self.pipes.keys()
        payloads = self.server_proxy.initialize_clients(cids)
        for cid, pipe in self.pipes.items():
            pipe.send(payloads[cid])

    def primary_upload(self):
        # server send models
        cids = self.pipes.keys()
        payloads = self.server_proxy.pre_primary_upload(cids=cids)
        for cid, pipe in self.pipes.items():
            pipe.send(payloads[cid])
        sprint('debug', f'The server have sent the untrained sub-model parameters for the clients')

    def primary_aggregate(self):
        # server receive trained model
        received_payloads = {}
        for cid, pipe in self.pipes.items():
            received_payloads[cid] = pipe.recv()
        sprint('debug', f"The server received trained_weights from the clients")
        # server aggregate trained model
        client_side_evolution = self.server_proxy.primary_aggregate(payloads=received_payloads)
        sprint('debug', f"The server have aggregated the parameters, losses, and metrics")
        return client_side_evolution

    def secondary_upload(self):
        # server send aggregated model
        cids = self.pipes.keys()
        payloads = self.server_proxy.pre_secondary_upload(cids=cids)
        for cid, pipe in self.pipes.items():
            pipe.send(payloads[cid])
        sprint('debug', f'The server have sent the trained global parameters for the clients')

    def secondary_aggregate(self):
        # server receive trained model
        received_payloads = {}
        for cid, pipe in self.pipes.items():
            received_payloads[cid] = pipe.recv()
        sprint('debug', f"The server received losses and the metrics from the clients")
        # server aggregate trained model
        server_side_evolution = self.server_proxy.secondary_aggregate(payloads=received_payloads)
        sprint('debug', "The server aggregated the losses and the metrics")
        return server_side_evolution

    def final_upload(self):
        # server send final global model
        cids = self.pipes.keys()
        payloads = self.server_proxy.pre_final_upload(cids=cids)
        for cid, pipe in self.pipes.items():
            pipe.send(payloads[cid])
        sprint('debug', f'The server have sent the final global parameters for the clients')

    def final_aggregate(self):
        # server receive trained model
        received_payloads = {}
        for cid, pipe in self.pipes.items():
            received_payloads[cid] = pipe.recv()
        sprint('debug', f"The server received losses and the metrics from the clients")
        # server aggregate trained model
        server_side_evolution = self.server_proxy.final_aggregate(payloads=received_payloads)
        sprint('debug', "The server aggregated the losses and the metrics")
        return server_side_evolution
