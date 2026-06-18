"""
BaseFL_public Server
"""
import copy
import csv
import numpy as np
import torch
from torch.optim import *
import os
from collections import defaultdict
import random
from tqdm import tqdm
import torch.nn as nn

from utils.train_utils import evaluate_model
from trainer.BaseFL.server import Server as Base_Server
from utils.nets import TwinBranchNets, ConditionalGenerator, Classifier, ResNet, BasicBlock 
from utils.loss import Gen_DiversityLoss

class Server(Base_Server):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.feat_gen_noise_dim = kwargs.get('feat_gen_noise_dim', 128) 
        self.global_feat_gen_epochs = kwargs.get('global_feat_gen_epochs', 10) 
        self.feat_gen_optim_lr = kwargs.get('feat_gen_optim_lr', 1e-3)
        self.feat_gen_optim_name =  kwargs.get('feat_gen_optim', 'Adam')
        self.global_feat_gen = None
        self.feat_gen_optimizer = None
        self.feat_gen_div_beta =  kwargs.get('feat_gen_div_beta', 1.0)

        self.feature_extractor = None
        self.optim_name =  kwargs.get('optim', 'Adam')
        self.optim_lr = kwargs.get('optim_lr', 1e-3)
        self.optimizer_fe = None
        
        self.diversity_loss = Gen_DiversityLoss(metric='l1')

    def distribute_model(self):
        pass

    def evaluate_private(self):
        pass
    
    def aggregate(self):
        super().aggregate()

        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            self.train_global_inference_model()
            self.test_global_inference_model()

    def train_global_inference_model(self):
        """reconstruct feature extractor to get a generic model"""

        if not hasattr(self, 'mse_loss_fn'):
            self.mse_loss_fn = nn.MSELoss()

        all_global_ids = set()
        for d_name, mapping in self.local_id_to_global_id.items():
            for l_id, g_id in mapping.items():
                all_global_ids.add(g_id)
        all_global_ids = list(all_global_ids)
        num_global_classes = len(all_global_ids)

        if num_global_classes == 0:
            self.logger.log("Warning: No mappings found via local_id_to_global_id. Skipping global model training.")
            return

        if self.model is None:
            self.logger.log(f"Initializing Global Inference Model with {num_global_classes} classes...")
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_global_classes, global_dim=256)
            
            optim_name = self.exp_conf.get('global_model_optim', 'Adam')
            optim_lr = self.exp_conf.get('global_model_optim_lr', 1e-3)
            self.global_model_optimizer = eval(optim_name)(self.model.parameters(), optim_lr)

        self.model.train()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        samples_per_class = self.exp_conf.get('global_samples_per_class', 64)
        batch_size = self.exp_conf.get('batch_size', 64)

        all_images = []
        all_labels = []
        samples_per_class = self.exp_conf.get('global_samples_per_class', 64)
        batch_size = self.exp_conf.get('batch_size', 64)

        for d_name, loader in self.train_loader.items():
            mapping = self.local_id_to_global_id.get(d_name, {})
            label_count = defaultdict(int)
            
            for x, y in loader:
                for i in range(len(y)):
                    lbl = int(y[i])
                    if lbl in mapping and label_count[lbl] < samples_per_class:
                        label_count[lbl] += 1
                        all_images.append(x[i])
                        all_labels.append(mapping[lbl])
                if all(label_count[l] >= samples_per_class for l in mapping.keys()):
                    break

        dataset_x = torch.stack(all_images)
        dataset_y = torch.tensor(all_labels, dtype=torch.long)
        
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(dataset_x, dataset_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(self.global_model_epochs), colour="green"):
            # all_images = []
            # all_labels = []
            
            # for d_name, loader in self.train_loader.items():
            #     mapping = self.local_id_to_global_id.get(d_name, {})
            #     label_count = defaultdict(int)
                
            #     for x, y in loader:
            #         for i in range(len(y)):
            #             lbl = int(y[i])
            #             if lbl in mapping and label_count[lbl] < samples_per_class:
            #                 label_count[lbl] += 1
            #                 all_images.append(x[i])
            #                 all_labels.append(mapping[lbl])
                    
            #         if all(label_count[l] >= samples_per_class for l in mapping.keys()):
            #             break

            # combined = list(zip(all_images, all_labels))
            # random.shuffle(combined)
            # all_images_shuffled, all_labels_shuffled = zip(*combined)

            # for i in range(0, len(combined), batch_size):
            #     batch_x = torch.stack(all_images_shuffled[i:i+batch_size]).to(self.device)
            #     batch_y_global = torch.tensor(all_labels_shuffled[i:i+batch_size], dtype=torch.long).to(self.device)

            #     feat = self.model.feature_extractor(batch_x)
            #     feat = torch.flatten(feat, 1)
            #     z_ = self.model.adapter(feat)
            #     logits = self.model.classifier(z_)

            #     loss_ce = criterion(logits, batch_y_global)
            #     loss = loss_ce

            #     self.global_model_optimizer.zero_grad()
            #     loss.backward()
            #     self.global_model_optimizer.step()

            epoch_loss = 0.0
            
            for batch_x, batch_y_global in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y_global = batch_y_global.to(self.device)

                feat = self.model.feature_extractor(batch_x)
                feat = torch.flatten(feat, 1)
                z_ = self.model.adapter(feat)
                logits = self.model.classifier(z_)

                loss_ce = criterion(logits, batch_y_global)
                
                self.global_model_optimizer.zero_grad()
                loss_ce.backward()
                self.global_model_optimizer.step()
                
                epoch_loss += loss_ce.item()
                
            #self.logger.log(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")
   
    def save_model(self, fname='checkpoints.pth'):
        self.logger.log("Saving BaseFL Public Dataset Server Checkpoints ...")

        client_label_distributions = {}
        for client in self.clients:
            unique_labels = set()
            for _, labels in client.train_loader:
                unique_labels.update(labels.tolist())
            client_label_distributions[client.id] = list(unique_labels)

        checkpoint = {
            'client_label_distributions': client_label_distributions,
            'global_registry': self.local_id_to_global_id,
            'label_space_meta': self.label_space_meta,
            'global_feature_dim': self.global_feature_dim,
            'exp_conf': self.exp_conf,
            'args': {
                'num_train_mnist': self.args.num_train_mnist,
                'num_train_emnist': self.args.num_train_emnist,
                'num_train_fashionmnist': self.args.num_train_fashionmnist,
                'num_train_cifar10': self.args.num_train_cifar10,
                'num_train_cifar100': self.args.num_train_cifar100,
                'num_new_clients': self.args.num_new_clients,
                'seed': self.args.seed,
                'device': str(self.args.device),
                'algorithm': self.args.algorithm,
            },
        }

        config_path = os.path.join(self.logger.log_dir, 'config_' + fname)
        torch.save(checkpoint, config_path)

        if self.model is not None:
            global_model_path = os.path.join(self.logger.log_dir, 'server_global_model.pth')
            torch.save(self.model.state_dict(), global_model_path)

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)
        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All clients saved in {clients_dir}/")