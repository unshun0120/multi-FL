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
import torch.nn.functional as F
from collections import defaultdict

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
        self.logger.log("[Server] Aggregating by Dataset Name ...")
        groups = defaultdict(list)
        for client in self.selected_clients:
            d_name = client.dataset_name  
            client.group_name = d_name
            groups[d_name].append(client)

            if d_name not in self.label_space_meta:
                self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} datasets)...") 

        if (self.glob_iter + 1) == self.start_mapping_epoch: 
            if self.args.label_mapping == 'class_name':
                self.class_name_label_mapping()
            elif self.args.label_mapping == 'real_image':
                self.real_img_label_mapping()
            elif self.args.label_mapping == 'independent':
                self.independent_label_mapping()
            elif self.args.label_mapping == 'identical':
                self.identical_label_mapping()

        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            self.train_global_inference_model()
            self.test_global_inference_model()

    def train_global_inference_model(self):
        """Train global inference model by client local feature distillation."""

        all_global_ids = set()
        for d_name, mapping in self.local_id_to_global_id.items():
            for l_id, g_id in mapping.items():
                all_global_ids.add(g_id)

        all_global_ids = sorted(list(all_global_ids))
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

        samples_per_class = self.exp_conf.get('global_samples_per_class', 64)
        batch_size = self.exp_conf.get('batch_size', 64)
        feature_kd_weight = self.exp_conf.get('feature_kd_weight', 1.0)

        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()

        dataset_teachers = defaultdict(list)
        for client in self.clients:
            d_name = client.dataset_name
            client.model.to(self.device)
            client.model.eval()
            dataset_teachers[d_name].append(client.model)

        def get_local_feature(model, x):
            if hasattr(model, "feature_extractor") and hasattr(model, "adapter"):
                feat = model.feature_extractor(x)
                feat = torch.flatten(feat, 1)
                z = model.adapter(feat)
                return z

            out = model(x)
            if isinstance(out, tuple):
                return out[0]
            return out

        all_images = []
        all_teacher_features = []
        all_hard_labels = []

        for d_name, loader in self.train_loader.items():
            mapping = self.local_id_to_global_id.get(d_name, {})
            teachers = dataset_teachers.get(d_name, [])

            if len(mapping) == 0 or len(teachers) == 0:
                continue

            label_count = defaultdict(int)

            for x, y in loader:
                selected_x = []
                selected_y_global = []

                for i in range(len(y)):
                    lbl = int(y[i])

                    if lbl in mapping and label_count[lbl] < samples_per_class:
                        label_count[lbl] += 1
                        selected_x.append(x[i])
                        selected_y_global.append(mapping[lbl])

                if len(selected_x) == 0:
                    if all(label_count[l] >= samples_per_class for l in mapping.keys()):
                        break
                    continue

                selected_x = torch.stack(selected_x).to(self.device)
                selected_y_global = torch.tensor(selected_y_global, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    teacher_features = []

                    for teacher in teachers:
                        teacher_feat = get_local_feature(teacher, selected_x)

                        if teacher_feat.dim() > 2:
                            teacher_feat = torch.flatten(teacher_feat, 1)

                        teacher_features.append(teacher_feat)

                    teacher_features = torch.stack(teacher_features, dim=0).mean(dim=0)
                    teacher_features = F.normalize(teacher_features, dim=1)

                all_images.append(selected_x.detach().cpu())
                all_teacher_features.append(teacher_features.detach().cpu())
                all_hard_labels.append(selected_y_global.detach().cpu())

                if all(label_count[l] >= samples_per_class for l in mapping.keys()):
                    break

        if len(all_images) == 0:
            self.logger.log("Warning: No feature distillation samples collected. Skipping global model training.")
            return

        dataset_x = torch.cat(all_images, dim=0)
        dataset_teacher_feat = torch.cat(all_teacher_features, dim=0)
        dataset_hard_y = torch.cat(all_hard_labels, dim=0)

        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(dataset_x, dataset_teacher_feat, dataset_hard_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(self.global_model_epochs), colour="green"):
            self.model.train()
            epoch_loss = 0.0

            for batch_x, batch_teacher_feat, batch_hard_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_teacher_feat = batch_teacher_feat.to(self.device)
                batch_hard_y = batch_hard_y.to(self.device)

                feat = self.model.feature_extractor(batch_x)
                feat = torch.flatten(feat, 1)
                z_ = self.model.adapter(feat)
                logits = self.model.classifier(z_)

                z_norm = F.normalize(z_, dim=1)

                loss_ce = criterion_ce(logits, batch_hard_y)
                loss_feature = criterion_mse(z_norm, batch_teacher_feat)

                loss = loss_ce + feature_kd_weight * loss_feature

                self.global_model_optimizer.zero_grad()
                loss.backward()
                self.global_model_optimizer.step()

                epoch_loss += loss.item()

            # self.logger.log(f"Global Feature KD Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")
    
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