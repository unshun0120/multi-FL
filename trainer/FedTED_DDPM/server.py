import copy
import csv
import numpy as np
import torch
from torch.optim import *
import os
from collections import OrderedDict, defaultdict
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

from utils.train_utils import evaluate_model
from trainer.BaseFL.server import Server as Base_Server
from utils.nets import TwinBranchNets, ConditionalGenerator, Classifier, ResNet, BasicBlock, ContextUnet, DDPM 
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

        self.global_ddpm_states = {}

    def aggregate(self):
        groups = defaultdict(list)
        for client in self.selected_clients:
            d_name = client.dataset_name  
            groups[d_name].append(client)

            if d_name not in self.label_space_meta:
                self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} datasets)...")

        for d_name, group_clients in groups.items():
            ddpm_msg_list = [
                (client.num_samples, client.ddpm.state_dict())
                for client in group_clients
            ]
            w_ddpm = self.aggregate_weights(ddpm_msg_list)
            self.global_ddpm_states[d_name] = w_ddpm

        if (self.glob_iter + 1) == self.start_mapping_epoch: 
            if self.index_matching == 'class_name':
                self.class_name_label_mapping()
            elif self.index_matching == 'real_image':
                self.real_img_label_mapping()
            elif self.index_matching == 'ddpm':
                self.ddpm_label_mapping()

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

            if client.dataset_name in self.global_ddpm_states:
                client.ddpm.load_state_dict(self.global_ddpm_states[client.dataset_name])

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

        generators = {}
        for d_name, state_dict in self.global_ddpm_states.items():
            num_local_classes = len(self.label_space_meta.get(d_name, []))
            if num_local_classes == 0: continue
            
            nn_model = ContextUnet(
                in_channels=self.exp_conf.get('channels', 3), 
                n_feat=self.exp_conf.get('n_feat', 64), 
                n_classes=num_local_classes
            ).to(self.device)
            gen = DDPM(
                nn_model=nn_model, betas=(1e-4, 0.02), n_T=1000, device=self.device, drop_prob=0.1
            ).to(self.device)
            gen.load_state_dict(state_dict)
            gen.eval()
            generators[d_name] = gen

        criterion = nn.CrossEntropyLoss()
        samples_per_class = self.exp_conf.get('global_samples_per_class', 64)
        batch_size = self.exp_conf.get('batch_size', 64)

        img_size = (self.exp_conf.get('channels', 3), self.exp_conf.get('img_size', 32), self.exp_conf.get('img_size', 32))
                
        dataset_x = []
        dataset_y = []
        
        for d_name, mapping in self.local_id_to_global_id.items():
            if d_name not in generators:
                continue
            gen = generators[d_name]

            for local_id, global_id in mapping.items():
                with torch.no_grad():
                    x_gen, _ = gen.sample(n_sample=self.global_samples_per_class, 
                                            size=img_size, 
                                            device=self.device, 
                                            guide_w=1.5, 
                                            label=local_id)

                    # x_gen = (x_gen + 1.0) / 2.0
                    # x_gen = torch.clamp(x_gen, 0.0, 1.0)
                    
                for i in range(self.global_samples_per_class):
                    dataset_x.append(x_gen[i].cpu())
                    dataset_y.append(global_id)

        if not dataset_x:
            self.logger.log("Warning: No synthetic data generated.")
            return

        dataset_x = torch.stack(dataset_x)
        dataset_y = torch.tensor(dataset_y, dtype=torch.long)
        
        train_dataset = TensorDataset(dataset_x, dataset_y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        for epoch in tqdm(range(self.global_model_epochs), colour="green", ncols=100):
            epoch_loss = 0
            
            for batch_x, batch_y_global in train_loader:
                batch_x, batch_y_global = batch_x.to(self.device), batch_y_global.to(self.device)
                #batch_x = normalize(batch_x)
                target_z = prox_z[batch_y_global]

                feat = self.model.feature_extractor(batch_x)
                feat = torch.flatten(feat, 1)
                z_ = self.model.adapter(feat)
                logits = self.model.classifier(z_)

                loss_mse = self.mse_loss_fn(z_, target_z)
                loss_ce = criterion(logits, batch_y_global)
                #loss = loss_mse + loss_ce
                loss = loss_ce

                self.global_model_optimizer.zero_grad()
                loss.backward()
                self.global_model_optimizer.step()
                
                epoch_loss += loss.item()
            
            self.logger.log(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")

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


    def aggregate_weights(self, weights_list):
            """
            FedAvg aggregation for Generator
            """
            total_samples = sum([w[0] for w in weights_list])
            avg_params = OrderedDict()
            
            for name in weights_list[0][1].keys():
                avg_params[name] = torch.zeros_like(weights_list[0][1][name], dtype=torch.float32)
                
                for num_samples, params in weights_list:
                    avg_params[name] += params[name] * (num_samples / total_samples)
                    
            return avg_params


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