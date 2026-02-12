import numpy as np
import torch
from torch.nn import *
from torch.optim import *
from tqdm import tqdm
from .client import Node
from collections import OrderedDict
import logging
import pandas as pd
import os
import time
from collections import defaultdict

from utils.train_utils import evaluate_model
from utils.nets import Classifier


class Server(Node):
    def __init__(self, clients, **kwargs):

        super(Server, self).__init__(**kwargs)
        self.algorithm_name = "BaseFL"

        self.sample_frac = kwargs.get('sample_frac', 1.0) 

        self.num_clients = len(clients)
        self.clients = clients
        self.registered_client_ids = [client.id for client in clients]
        self.selected_clients_ids = []
        self.selected_clients = []
        
        # Value: {'feature_extractor': state_dict, 'classifier': state_dict}
        self.global_models = {}

        self.global_rounds = kwargs.get('global_rounds', 100)

        self.metric_type = kwargs.get('metric_type', 'accuracy')
        self.metric = OrderedDict()

        self.p_acc, self.p_loss, self.g_acc, self.g_loss = 0, 0, 0, 0


    def run(self):
        """ The train process of FedAvg, in the child class of FedAvg server,
            override the corresponding method if the process is same.

        Args:
            rounds: total communication rounds
            epochs: local update epochs of each client
            test_interval: per x round, test performance
            verbose: 0, print nothing, 1 print acc
            save_ckpt: bool, if save ckpt in each test interval. default false.
        """
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log(f"Start {self.global_rounds} rounds training by {self.algorithm_name}")

        self.init_metric()

        for r in range(self.global_rounds):
            # # Debug: did server w change? - before
            # server_w_before = copy.deepcopy(self.model.state_dict())
            # client0_w_before = copy.deepcopy(self.clients[0].model.state_dict())

            self.glob_iter = r

            # step 1. sample clients
            self.sample_clients()
            self.distribute_model()

            self.local_update()
            
            if (r + 1) % self.test_interval == 0:
                self.evaluate_private()

            # step 3. aggregate
            self.aggregate()
            if (r + 1) % self.test_interval == 0:
                # self.evaluate_generic()
                self.record_metric(r, self.p_acc, self.p_loss, self.g_acc, self.g_loss)

            if not self.args.no_save_model:
                os.makedirs(f'./ckpt/{self.glob_iter + 1}', exist_ok=True)
                torch.save(self.model.state_dict(), f'./ckpt/{self.glob_iter + 1}/server.pth')
                for c in self.selected_clients:
                    torch.save(c.model.state_dict(), f'./ckpt/{self.glob_iter + 1}/client_{c.id}.pth')

            # # Debug: did server w change? - after
            # server_w_after = self.model.state_dict()
            # client0_w_after = self.clients[0].model.state_dict()

        # save metric
        self.save_metric()


    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randomly
        num_sampled_clients = max(int(self.sample_frac * self.num_clients), 1)
        self.selected_clients_ids = sorted(np.random.choice(range(self.num_clients),
                                                            size=num_sampled_clients,
                                                            replace=False).tolist())
        
        self.logger.log(f'Selected client ids: {self.selected_clients_ids}')

        self.selected_clients = [self.clients[idx] for idx in self.selected_clients_ids]

        for client in self.selected_clients:
            client.glob_iter = self.glob_iter

    def distribute_model(self):
        for client in self.selected_clients:
            ls_id = str(client.class_name_set)
            
            if ls_id not in self.global_models:
                continue

            global_part = self.global_models[ls_id]

            if 'classifier' in global_part:
                client.model.classifier.load_state_dict(global_part['classifier'])

            if not self.heterogeneous and 'feature_extractor' in global_part:
                client.model.feature_extractor.load_state_dict(global_part['feature_extractor'])

    def local_update(self):
        self.logger.log(f"--- Round {self.glob_iter + 1} ---")
        for client in tqdm(self.selected_clients):
            client.update()
           
    def aggregate(self):
        groups = defaultdict(list)
        for client in self.selected_clients:
            ls_id = str(client.class_name_set)
            groups[ls_id].append(client)
            
            if ls_id not in self.label_space_meta:
                self.label_space_meta[ls_id] = client.class_name_set

            # print(ls_id)
            # e.g. ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            # ["Ankle boot", "Bag", "Coat", "Dress", "Pullover", "Sandal", "Shirt", "Sneaker", "T-shirt/top", "Trouser"]

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} label spaces)...") 
        
        for ls_id, group_clients in groups.items():
            if ls_id not in self.global_models:
                self.global_models[ls_id] = {}

            # aggregate clients generic classifier
            msg_list = [(client.num_samples, client.model.classifier.state_dict())
                        for client in group_clients]
            w_cls = self.avg_weights(msg_list)

            num_classes = w_cls['weight'].shape[0]
            input_dim = w_cls['weight'].shape[1]

            cls_model = Classifier(input_dim, num_classes).to(self.device)
            cls_model.load_state_dict(w_cls)
            cls_model.eval()

            self.global_models[ls_id]['classifier'] = cls_model

            # if not heterogeneous, aggregate feature_extractor of clients
            # if not self.heterogeneous:
            #     msg_list = [(client.num_samples, client.model.feature_extractor.state_dict())
            #                 for client in group_clients]
            #     w_fe = self.avg_weights(msg_list)

            #     fe_model = copy.deepcopy(clients[0].model.feature_extractor)
            #     fe_model.load_state_dict(w_fe)
            #     fe_model.eval().to(self.device)
                
            #     self.global_models[ls_id]['feature_extractor'] = fe_model

            

    @staticmethod
    def avg_weights(nk_and_wk):
        """
        n_k_and_weights: [..., (n_k, w_k), ....], where n_k is the number of samples w_k is weight.
        """
        averaged_weights = OrderedDict()

        n_sum = sum([n_k for n_k, _ in nk_and_wk])
        for i, (n_k, w_k) in enumerate(nk_and_wk):
            for key in w_k.keys():
                averaged_weights[key] = n_k / n_sum * w_k[key] if i == 0 \
                    else averaged_weights[key] + n_k / n_sum * w_k[key]
        return averaged_weights
    

    """Metric the performance"""

    def init_metric(self):
        self.metric['round'] = []
        self.metric[f'private_{self.metric_type}'] = []
        self.metric['private_loss'] = []
        self.metric[f'general_{self.metric_type}'] = []
        self.metric['general_loss'] = []

    def record_metric(self, r, private_accuracy, private_loss, general_accuracy, general_loss):
        self.metric['round'].append(r)
        self.metric[f'private_{self.metric_type}'].append(private_accuracy)
        self.metric['private_loss'].append(private_loss)
        self.metric[f'general_{self.metric_type}'].append(general_accuracy)
        self.metric['general_loss'].append(general_loss)

        self.logger.log(f"\n round {r:0>3d}, \n"
                      f"\t \t private_loss:{private_loss:.4f}, \n"
                      f"\t \t private_{self.metric_type}:{private_accuracy:.4f} \n"
                      f"\t \t general_loss:{general_loss:.4f},  \n"
                      f"\t \t general_{self.metric_type}:{general_accuracy:.4f} \n")

    def evaluate_generic(self):
        acc_list, loss_list = [], []
        g_acc, g_loss = evaluate_model(self.model, self.test_loader, self.loss_fn,
                                        self.metric_type, self.device)
        acc_list.append(g_acc)
        loss_list.append(g_loss)
        self.g_acc, self.g_loss = np.mean(acc_list), np.mean(loss_list)

    def evaluate_private(self):
        acc_list, loss_list = [], []
        for client in self.selected_clients:
            p_acc, p_loss = evaluate_model(client.model, client.test_loader, self.loss_fn,
                                           self.metric_type, self.device)
            client.round_test_acc = p_acc
            acc_list.append(p_acc)
            loss_list.append(p_loss)
            self.logger.log(f"Round {self.glob_iter + 1} | Client {client.id} | Model: ({client.model_name}) | Loss: {client.round_train_loss:.4f} | Acc: {client.round_test_acc*100:.2f}%")

        self.p_acc, self.p_loss = np.mean(acc_list), np.mean(loss_list)

    def save_metric(self):
        pd.DataFrame(self.metric).to_csv(
            os.path.join(self.logger.get_log_dir(), "metric.csv"), index=False)




