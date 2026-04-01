from ..BaseFL.client import Client as BaseClient


class Client(BaseClient):
    """Here, client is only for test personal performance"""

    def __init__(self, **exp_conf):
        super(Client, self).__init__(**exp_conf)