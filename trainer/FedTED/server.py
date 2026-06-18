"""
FedTED Server
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

    def aggregate(self):
        super().aggregate()

        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            self.train_generator()
            self.train_global_inference_model()
            self.test_global_inference_model()


    def distribute_model(self):
        gen_state_dict = None
        prox_z = None
        
        if self.global_feat_gen is not None:
            gen_state_dict = self.global_feat_gen.state_dict()
            
            all_global_ids = set()
            for mapping in self.local_id_to_global_id.values():
                for g_id in mapping.values():
                    all_global_ids.add(g_id)
            if len(all_global_ids) > 0:
                num_global_classes = max(all_global_ids) + 1
                prox_z_tensor, _ = self.gen_prox_data(num_global_classes)
                prox_z = prox_z_tensor.detach().clone().cpu()

        for client in self.selected_clients:
            if client.dataset_name in self.local_id_to_global_id:
                client.local_id_to_global_id = self.local_id_to_global_id[client.dataset_name]

            if self.global_feat_gen is not None:
                if getattr(client, 'global_feat_gen', None) is None:
                    client.global_feat_gen = copy.deepcopy(self.global_feat_gen)
                else:
                    client.global_feat_gen.load_state_dict(gen_state_dict)
                
                if prox_z is not None:
                    client.prox_z = prox_z

            ls_id = str(client.class_name_set)
            
            if ls_id not in self.global_models:
                continue

            global_part = self.global_models[ls_id]

            if 'classifier' in global_part:
                client.model.classifier.load_state_dict(global_part['classifier'].state_dict())


    def train_generator(self):
        all_global_ids = set()
        for d_name, mapping in self.local_id_to_global_id.items():
            for l_id, g_id in mapping.items():
                all_global_ids.add(g_id)

        if len(all_global_ids) == 0:
            self.logger.log("Warning: No mappings found. Applying identical label mapping for single dataset...")
            self.identical_label_mapping()
            for d_name, mapping in self.local_id_to_global_id.items():
                for l_id, g_id in mapping.items():
                    all_global_ids.add(g_id)

        all_global_ids = list(all_global_ids)

        global_num_classes = max(all_global_ids) + 1
        if self.global_feat_gen is None:
            self.global_feat_gen = ConditionalGenerator(
                num_global_classes=global_num_classes, 
                noise_dim=self.feat_gen_noise_dim,
                output_dim=self.global_feature_dim  
            ).to(self.device)
            self.feat_gen_optimizer = eval(self.feat_gen_optim_name)(self.global_feat_gen.parameters(), self.feat_gen_optim_lr)

        self.global_feat_gen.train()

        for client in self.selected_clients:
            client.model.classifier.eval()
            client.model.classifier.to(self.device)

        all_classifiers = {}
        for ls_id, model_dict in self.global_models.items():
            if 'classifier' in model_dict:
                all_classifiers[ls_id] = model_dict['classifier']

        criterion = nn.CrossEntropyLoss()

        self.logger.log("[Server] Training Feature-Based Global Generator...")
        for epoch in tqdm(range(self.global_feat_gen_epochs), colour="blue"):
            epoch_loss = 0

            batch_ids = np.random.choice(all_global_ids, self.batch_size)  
            labels_input = torch.tensor(batch_ids, dtype=torch.int64).to(self.device)

            z = torch.randn(self.batch_size, self.feat_gen_noise_dim).to(self.device)
            gen_feat = self.global_feat_gen(z, labels_input)

            div_loss = self.diversity_loss(gen_feat, z)

            cls_loss = 0.0
            valid_client_count = 0

            for client in self.selected_clients:
                global_to_local = {}
                d_name = client.dataset_name
                if d_name in self.local_id_to_global_id:
                    for l_id, g_id in self.local_id_to_global_id[d_name].items():
                        global_to_local[g_id] = l_id
                
                target_list = []
                for gid in batch_ids:
                    if gid in global_to_local:
                        target_list.append(global_to_local[gid])
                    else:
                        target_list.append(-1)
                
                target_tensor = torch.tensor(target_list, dtype=torch.long).to(self.device)
                mask = (target_tensor != -1) 

                if mask.any():
                    valid_feat = gen_feat[mask]
                    valid_targets = target_tensor[mask]
                    
                    logits = client.model.classifier(valid_feat)
                    
                    loss = criterion(logits, valid_targets)
                    cls_loss += loss
                    valid_client_count += 1

            if valid_client_count > 0:
                cls_loss /= valid_client_count

            total_loss = self.feat_gen_div_beta * div_loss + cls_loss
            
            self.feat_gen_optimizer.zero_grad()
            total_loss.backward()
            self.feat_gen_optimizer.step()


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

        prox_z, prox_y = self.gen_prox_data(num_global_classes)
        prox_z = prox_z.to(self.device).detach()

        if self.model is None:
            self.logger.log(f"Initializing Global Inference Model with {num_global_classes} classes...")
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_global_classes, global_dim=256)
            params = list(self.model.feature_extractor.parameters()) + \
                     list(self.model.adapter.parameters()) + \
                     list(self.model.classifier.parameters())
            self.global_model_optimizer = eval(self.global_model_optim_name)(params, self.global_model_optim_lr)

            self.feature_extractor = self.model.feature_extractor
            self.optimizer_fe = eval(self.optim_name)(
                filter(lambda p: p.requires_grad, self.model.feature_extractor.parameters()),
                self.optim_lr)

        self.model.train()
        self.model.to(self.device)

        # mixed_batches = []
        # for d_name, loader in self.train_loader.items():
        #     mapping = self.local_id_to_global_id.get(d_name, {})
        #     for x, y in loader:
        #         y_global = torch.tensor([mapping[int(lbl)] for lbl in y], dtype=torch.long)
        #         mixed_batches.append((x.cpu(), y_global.cpu()))

        # self.logger.log(f"Mixed total {len(mixed_batches)} batches across all datasets.")

        criterion = nn.CrossEntropyLoss()
        samples_per_class = self.exp_conf.get('global_samples_per_class', 64)
        batch_size = self.exp_conf.get('batch_size', 64)

        for epoch in tqdm(range(self.global_model_epochs), colour="green"):
            all_images = []
            all_labels = []
            
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

            combined = list(zip(all_images, all_labels))
            random.shuffle(combined)
            all_images_shuffled, all_labels_shuffled = zip(*combined)

            for i in range(0, len(combined), batch_size):
                batch_x = torch.stack(all_images_shuffled[i:i+batch_size]).to(self.device)
                batch_y_global = torch.tensor(all_labels_shuffled[i:i+batch_size], dtype=torch.long).to(self.device)

                target_z = prox_z[batch_y_global]

                feat = self.model.feature_extractor(batch_x)
                feat = torch.flatten(feat, 1)
                z_ = self.model.adapter(feat)
                logits = self.model.classifier(z_)

                loss_mse = self.mse_loss_fn(z_, target_z)
                loss_ce = criterion(logits, batch_y_global)
                loss = loss_mse + loss_ce

                self.global_model_optimizer.zero_grad()
                loss.backward()
                self.global_model_optimizer.step()

    def gen_prox_data(self, num_global_classes):
        prox_z = []
        prox_y = list(range(num_global_classes))

        samples_per_class = 100

        self.global_feat_gen.eval()

        for i in range(num_global_classes):
            y = torch.full((samples_per_class,), i, dtype=torch.long).to(self.device)
            z_noise = torch.randn(samples_per_class, self.feat_gen_noise_dim).to(self.device)
            
            with torch.no_grad():
                z = self.global_feat_gen(z_noise, y)
                
            class_prototype = z.mean(dim=0).detach().clone()
            prox_z.append(class_prototype)

        return torch.stack(prox_z, dim=0), torch.tensor(prox_y)


    def evaluate_private(self):
        # use the client-side model
        acc_list, loss_list = [], []
        dataset_accs = defaultdict(list)

        for client in self.selected_clients:
            # turn on the private mode
            client.model.use_twin = True
            p_acc = evaluate_model(client.model, client.test_loader, 
                                           self.metric_type, self.device)
            # turn off the private mode
            client.model.use_twin = False
            client.round_test_acc = p_acc
            acc_list.append(p_acc)

            dataset_accs[client.dataset_name].append(p_acc)

            self.logger.log(f"Round {self.glob_iter + 1} | Client {client.id} | Model: ({client.model_name}) | Acc: {client.round_test_acc*100:.2f}%")

        self.p_acc = np.mean(acc_list)

        for d_name, accs in dataset_accs.items():
            self.dataset_acc_history[d_name].append(np.mean(accs) * 100.0)


    def save_model(self, fname='checkpoints.pth'):
        self.logger.log("Saving UDON Server Checkpoints ...")

        client_label_distributions = {}
        for client in self.clients:
            unique_labels = set()
            for _, labels in client.train_loader:
                unique_labels.update(labels.tolist())
            client_label_distributions[client.id] = list(unique_labels)

        checkpoint = {
            'generator': self.global_feat_gen.state_dict() if hasattr(self, 'global_feat_gen') else None,
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