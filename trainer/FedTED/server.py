import copy
import numpy as np
import torch
from torch.optim import *
import os
from collections import defaultdict
import random
from tqdm import tqdm

from utils.train_utils import evaluate_model
from trainer.BaseFL.server import Server as Base_Server
from utils.nets import TwinBranchNets
from utils.nets import ConditionalGenerator, Classifier


class Server(Base_Server):
    def __init__(self, fine_tune=True, rebuild_loader=None, **kwargs):
        super(Server, self).__init__(**kwargs)
        
        self.algorithm_name = "FedTED"

        self.heterogeneous = kwargs.get('heterogeneous', False) 

        # generator trainer
        self.feat_gen_noise_dim = kwargs.get('feat_gen_noise_dim', 128)

        self.global_feature_gen = ConditionalGenerator(
            num_global_classes=len(self.local_label_to_global_id),
            noise_dim=self.feat_gen_noise_dim,
            output_dim=self.global_feature_dim 
        ).to(self.device)

        self.global_feat_gen_epochs = kwargs.get('global_feat_gen_epochs', 10) 
        self.feat_gen_optim_lr = kwargs.get('feat_gen_optim_lr', 1e-3)
        self.feat_gen_optim_name =  kwargs.get('feat_gen_optim', 'Adam')
        self.feat_gen_optimizer = eval(self.feat_gen_optim_name)(self.global_feature_gen.parameters(), self.feat_gen_optim_lr)


        # rebuilder trainer
        self.feature_extractor = self.model.feature_extractor
        self.optimizer_fe = eval(self.opt_name)(
            filter(lambda p: p.requires_grad, self.model.feature_extractor.parameters()),
            **self.optim_kwargs)
        
        self.rebuild_generic_model_epochs = kwargs.get('rebuild_generic_model_epochs', 15)

        self.gen_div_beta =  kwargs.get('gen_div_beta', 1.0)

        self.mse_loss_fn = torch.nn.MSELoss()
        self.fine_tune = fine_tune

        self.rebuild_loader = self.train_loader

    def distribute_model(self):
        gen_state_dict = None
        if self.global_feature_gen is not None:
            gen_state_dict = self.global_feature_gen.state_dict()

        for client in self.selected_clients:
            ls_id = str(client.class_name_set)
            
            if ls_id not in self.global_models:
                continue

            global_part = self.global_models[ls_id]

            if 'classifier' in global_part:
                client.model.classifier.load_state_dict(global_part['classifier'].state_dict())

            if gen_state_dict is not None:
                if hasattr(client, 'global_feature_gen') and client.global_feature_gen is not None:
                    client.global_feature_gen.load_state_dict(gen_state_dict)


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

        # update generator
        self.train_generator()

        #self.rebuild_generic()

    def train_generator(self):
        self.global_feature_gen.train()

        for client in self.selected_clients:
            client.model.classifier.eval()
            client.model.classifier.to(self.device)

        all_classifiers = {}
        for ls_id, model_dict in self.global_models.items():
            if 'classifier' in model_dict:
                all_classifiers[ls_id] = model_dict['classifier']

        all_global_ids = list(self.global_id_to_local_label.keys())

        self.logger.log("[Server] Training Feature-Based Global Generator...")
        for epoch in tqdm(range(self.global_feat_gen_epochs), colour="blue"):
            epoch_loss = 0
            random.shuffle(all_global_ids)

            batch_ids = np.random.choice(all_global_ids, self.batch_size)  
            labels_input = torch.tensor(batch_ids, dtype=torch.int64).to(self.device)

            z = torch.randn(self.batch_size, self.feat_gen_noise_dim).to(self.device)
            gen_feat = self.global_feature_gen(z, labels_input)

            div_loss = self.diversity_loss(gen_feat, z)

            cls_loss = 0.0
            valid_client_count = 0

            for client in self.selected_clients:
                client_class_names = client.class_name_set
                valid_indices = []
                local_targets = []

                for i, gid in enumerate(batch_ids):
                    g_name = self.global_id_to_local_label[gid]
                    if g_name in client_class_names:
                        valid_indices.append(i)
                        local_targets.append(client_class_names.index(g_name))

                if len(valid_indices) > 0:
                    curr_feat = gen_feat[valid_indices]
                    curr_target = torch.tensor(local_targets, dtype=torch.long).to(self.device)
                    
                    logits = client.model.classifier(curr_feat)
                    
                    loss = self.loss_fn(logits, curr_target)
                    cls_loss += loss
                    valid_client_count += 1

            if valid_client_count > 0:
                cls_loss /= valid_client_count

            total_loss = self.gen_div_beta * div_loss + cls_loss
            
            self.feat_gen_optimizer.zero_grad()
            total_loss.backward()
            self.feat_gen_optimizer.step()


    def rebuild_generic(self):
        """reconstruct feature extractor to get a generic model"""
        self.feature_extractor.train()
        self.feature_extractor.to(self.device)

        # get little batch of generic server or client
        x, y = next(iter(self.rebuild_loader))
        if y.size(0) <= 1:
            x, y = next(iter(self.rebuild_loader))

        # z, _ = self.generator(y)
        prox_z, prox_y = self.gen_prox_data()
        batch_z = []
        for j in range(y.size(0)):
            batch_z.append(prox_z[int(y[j])])
        z = torch.stack(batch_z, dim=0)

        for epoch in range(self.rebuild_generic_model_epochs):
            x, z = x.to(self.device), z.to(self.device)

            z_ = self.feature_extractor(x)

            loss = self.mse_loss_fn(z_, z)

            self.optimizer_fe.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_fe.step()

    def gen_prox_data(self):
        porx_z = [0.] * self.num_classes
        porx_y = list(range(self.num_classes))

        batch_labels = np.random.choice(self.num_classes, self.batch_size * 100)
        y = torch.tensor(batch_labels, dtype=torch.int64)
        z, _ = self.global_feature_gen(y)

        for i in range(self.num_classes):
            idx = torch.nonzero(y == i).view(-1)
            if len(idx) > 0:
                porx_z[i] += (z[idx].sum(dim=0) / len(idx))

        return torch.stack(porx_z, dim=0), torch.tensor(porx_y)

    def evaluate_generic(self):
        # use the rebuild generic model
        self.g_acc, self.g_loss = evaluate_model(self.model, self.test_loader, self.loss_fn,
                                                 self.metric_type, self.device)

    def evaluate_private(self):
        # use the client-side model
        acc_list, loss_list = [], []
        for client in self.selected_clients:
            # turn on the private mode
            client.model.use_twin = True
            g_acc, g_loss = evaluate_model(client.model, client.test_loader, self.loss_fn,
                                           self.metric_type, self.device)
            # turn off the private mode
            client.model.use_twin = False
            acc_list.append(g_acc)
            loss_list.append(g_loss)
        self.p_acc, self.p_loss = np.mean(acc_list), np.mean(loss_list)
        return self.p_acc, self.p_loss
