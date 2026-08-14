import torch
from torch.nn import *
from torch.optim import *

from utils.train_utils import train_model, evaluate_model

class Node:
    """ A computation node, could be clients, servers or any computed devices.

    It can be understood as driver for devices.
    """

    def __init__(self, node_id, args, dataset_name, train_loader, test_loader, 
                 model, class_name_set, model_name, logger, **exp_conf):
        # args
        self.args = args
        self.device = args.device
        self.algorithm = args.algorithm

        # input parameter
        self.id = node_id
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.model_name = model_name
        self.class_name_set = class_name_set
        self.logger = logger

        # experiment config
        self.heterogeneous = exp_conf.get('heterogeneous', False)
        self.dirichlet_alpha = exp_conf.get('dirichlet_alpha', 1.0)
        self.batch_size = exp_conf.get('batch_size', 64)
        self.test_interval = exp_conf.get('test_interval', 1)
        self.global_feature_dim = exp_conf.get('global_feature_dim', 256)
        #self.start_mapping_epoch = exp_conf.get('start_mapping_epoch', 1)
        self.start_mapping_epoch = args.start_mapping_epoch

        # initial variables
        self.label_space_meta =  {} # { 'ls_id': ['dog', 'cat', ...] }
        self.local_id_to_global_id = {}
        self.glob_iter = 0
        self.round_train_loss = 0.0
        self.round_test_acc = 0.0
    

class Client(Node):
    """BaseFL client"""

    def __init__(self, **exp_conf):
        super(Client, self).__init__(**exp_conf)
        self.local_epochs = exp_conf.get('local_epochs', 0) 
        
        # local loss
        self.local_loss_name = exp_conf.get('loss', 'CrossEntropyLoss')
        self.local_loss_fn = eval(self.local_loss_name)()

        # local optimizer        
        self.local_lr = exp_conf.get('local_lr', 1e-3)
        self.local_optim_name =  exp_conf.get('local_optim', 'Adam')
        self.local_optimizer = eval(self.local_optim_name)(self.model.parameters(), self.local_lr)

        self.num_samples = len(self.train_loader.dataset)
        self.local_num_classes = len(self.class_name_set)


    def update(self):
        """train node's model by local train dataset"""
        self.round_train_loss = train_model(self.model, self.train_loader, self.local_optimizer,
                                            self.local_loss_fn, self.local_epochs, self.device)
        

    def evaluate(self, metric_type='accuracy'):
        """evaluate node's model by local test dataset
        :return correct, test_loss
        """
        return evaluate_model(self.model, self.test_loader, self.local_loss_fn,
                              metric_type, self.device)