from ..BaseFL.client import Client as BaseClient
from utils.loss import UDONLoss

class Client(BaseClient):
    """Here, client is only for test personal performance"""

    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)
        self.loss_fn = UDONLoss()