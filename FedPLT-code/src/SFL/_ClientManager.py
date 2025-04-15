from src.SFL.Tools import _Pipe
from src.SFL._Client import _Client


class _ClientManager:
    def __init__(self):
        self._clients = {}
        self._pipes = {}

    def create_client(self, cid, client_proxy):
        pipe = _Pipe()
        client = _Client(cid, pipe, client_proxy)
        self._clients[cid] = client
        self._pipes[cid] = pipe

    def get_client(self, cid):
        return self._clients[cid]

    def get_pipe(self, cid):
        return self._pipes[cid]
