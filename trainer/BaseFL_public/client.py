from ..BaseFL.client import Client as BaseClient

class Client(BaseClient):
    """Here, client is only for test personal performance"""

    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)

    def update(self):
        pass

    def evaluate(self):
        pass