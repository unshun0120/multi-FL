import torch
from torch.nn import *
from torch.optim import *
import os

from utils.loss import VanillaKDLoss, Gen_DiversityLoss
from utils.train_utils import train_model, evaluate_model

class Node:
    """ A computation node, could be clients, servers or any computed devices.

    It can be understood as driver for devices.
    """

    def __init__(self, node_id, args, dataset_name, train_loader, test_loader, model, class_name_set, model_name, global_registry, logger, **kwargs):
        self.args = args
        self.device = args.device
        
        self.id = node_id
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_samples = len(train_loader.dataset)
        self.model = model
        self.class_name_set = class_name_set
        self.model_name = model_name
        self.logger = logger

        self.local_label_to_global_id = global_registry # {'dog': 0, 'cat': 1, ...}
        self.global_id_to_local_label = {
            v: k for k, v in self.local_label_to_global_id.items()} # {0: 'dog', 1: 'cat', ...}
        
        if class_name_set is not None:
            self.local_int_to_global_int = {}
            for local_idx, name in enumerate(self.class_name_set):
                if name in global_registry:
                    global_id = global_registry[name]
                    self.local_int_to_global_int[local_idx] = global_id
        
        self.num_global_classes = len(self.local_label_to_global_id)

        self.label_space_meta =  {} # { 'ls_id': ['dog', 'cat', ...] }

        self.heterogeneous = kwargs.get('heterogeneous', False)
        self.batch_size = kwargs.get('batch_size', 64)
        self.loss = kwargs.get('loss', 'CrossEntropyLoss')
        self.loss_fn = eval(self.loss)()
        self.test_interval = kwargs.get('test_interval', 1)

        if self.model is not None:
            self.optim_kwargs = kwargs.get('optim_kwargs', {'lr': 1e-3})
            self.opt_name =  kwargs.get('optim', 'Adam')
            self.optimizer = eval(self.opt_name)(self.model.parameters(), **self.optim_kwargs)
        else:
            self.optimizer = None

        self.global_feature_dim = kwargs.get('global_feature_dim', 256)

        self.glob_iter = 0
        self.round_train_loss = 0.0
        self.round_test_acc = 0.0

        self.distill_temperature = kwargs.get('distill_temperature', 20)
        self.kd_loss_fn = VanillaKDLoss(temperature=self.distill_temperature)

        self.diversity_loss = Gen_DiversityLoss(metric='l1')
        

    def update(self):
        """train node's model by local train dataset"""
        self.round_train_loss = train_model(self.model, self.train_loader, self.optimizer,
                    self.loss_fn, self.local_epochs, self.device)
        
    def evaluate(self, metric_type='accuracy'):
        """evaluate node's model by local test dataset
        :return correct, test_loss
        """
        return evaluate_model(self.model, self.test_loader, self.loss_fn,
                              metric_type, self.device, self.test_interval)

    def save(self, fname='model.pt'):
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, fname))

    def load(self, fname='model.pt'):
        state_dict = torch.load(os.path.join(self.log_dir, fname))
        self.model.load_state_dict(state_dict)


class Client(Node):
    """BaseFL client"""

    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)
        self.local_epochs = kwargs.get('local_epochs', 1) 
        self.local_num_classes = len(self.class_name_set)