import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..BaseFL.client import Client as BaseClient
from utils.nets import ConditionalGenerator


class Client(BaseClient):
    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)
