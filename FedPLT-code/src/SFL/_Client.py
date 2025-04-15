from src.SFL.Tools import sprint


class _Client:
    def __init__(self, cid, pipe, client_proxy):
        super().__init__()
        self.cid = cid
        self.pipe = pipe

        self.client_proxy = client_proxy

    def initialization(self):
        payload = self.pipe.recv()
        self.client_proxy.configure(payload)

    def train_validate_local_model(self):
        # clients receive models
        # while not self.pipe.poll():
        #     pass
        payload = self.pipe.recv()
        sprint('debug', "Client {} received untrained sub-model parameters from the server".format(self.cid + 1))

        # clients train and validate the models
        sprint('debug', f'client {self.cid + 1} is fitting')
        payload = self.client_proxy.fit(payload)

        # clients send trained model
        # clients send metrics and loses
        self.pipe.send(payload)
        sprint('debug',
               f"client {self.cid + 1} sent  trained sub-model parameters, along with losses and metrics to the server")

    def validate_global_model(self):
        # client receives aggregated model
        # while not self.pipe.poll():
        #     pass
        payload = self.pipe.recv()
        sprint('debug', f"Client {self.cid + 1} received trained global parameters from the server")

        # client validated aggregated model
        # client send losses and metrics
        sprint('debug', f'client {self.cid + 1} is validating')
        payload = self.client_proxy.val(payload)
        self.pipe.send(payload)
        sprint('debug', f"client {self.cid + 1} sent losses and metrics to the server")

    def test_global_model(self):
        # client receives aggregated model
        # while not self.pipe.poll():
        #     pass
        payload = self.pipe.recv()
        sprint('debug', f"Client {self.cid + 1} received final global parameters from the server")

        # client validated aggregated model
        # client send losses and metrics
        sprint('debug', f'client {self.cid + 1} is testing')
        payload = self.client_proxy.test(payload)
        self.pipe.send(payload)
        sprint('debug', f"client {self.cid + 1} sent losses and metrics to the server")
