"""
GeFL_DDPM_baseline_total Server
"""
import torch
import copy
from collections import OrderedDict, defaultdict
import os
import csv
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import random
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from utils.nets import ContextUnet, DDPM
# from utils.ddpm_nets import ContextUnet, DDPM
from label_mapping.label_mapping_utils import (
    label_mapping, evaluate_mapping_results, 
    feature_bi_direction_label_mapping, single_direction_label_mapping,
    get_gen_images, global_to_local_mapping, clear_image_caches,
    image_cosine_similarity_mapping,
)
from utils.nets import ResNet, BasicBlock

class Server(BaseServer):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.global_ddpm_states = {}

    def run(self):
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log(f"Start {self.global_rounds} rounds training by {self.algorithm}")

        for r in range(self.global_rounds):
            self.glob_iter = r

            self.sample_clients()
            self.distribute_model()
            self.local_update()

            # if (r + 1) % self.test_interval == 0:
            #     self.evaluate_private()

            self.aggregate()

            # if r+1 == 40:
            #     self.save_model() 

        # self.save_model()
        # plot_accuracy_curves(self.dataset_acc_history, self.logger.log_dir, self.args, self.global_rounds, self.dirichlet_alpha)


    def get_all_clients_averaged_features(self):
        dataset_features = {}
        for client in self.clients:
            d_id = client.dataset_name
            if d_id not in dataset_features:
                dataset_features[d_id] = {}
            
            client_avg_feats = client.get_avg_features()
            for lbl, feat in client_avg_feats.items():
                if lbl not in dataset_features[d_id]:
                    dataset_features[d_id][lbl] = []
                dataset_features[d_id][lbl].append(feat)
                
        for d_id in dataset_features:
            for lbl in dataset_features[d_id]:
                dataset_features[d_id][lbl] = torch.cat(dataset_features[d_id][lbl], dim=0)
                
        return dataset_features
    

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

            clear_image_caches()
            dataset_clients_dict = {}   
            active_datasets = []
            dataset_label_space_meta = {}
            
            for client in self.clients: 
                d_name = client.dataset_name
                
                if d_name not in dataset_clients_dict:
                    dataset_clients_dict[d_name] = []
                    active_datasets.append(d_name)
                    dataset_label_space_meta[d_name] = self.label_space_meta[d_name]
                
                dataset_clients_dict[d_name].append(client.model.to(self.device))
            
            active_datasets = list(dataset_clients_dict.keys())

            ddpm_dict = {}
            for d_name in active_datasets:
                if d_name in self.global_ddpm_states:
                    nn_model = ContextUnet(
                        in_channels=self.exp_conf.get('channels', 3), 
                        n_feat=self.exp_conf.get('n_feat', 64), 
                        n_classes=len(self.label_space_meta[d_name])
                    ).to(self.device)
                    
                    ddpm = DDPM(
                        nn_model=nn_model, 
                        betas=(1e-4, 0.02), 
                        n_T=1000, 
                        device=self.device, 
                        drop_prob=0.1
                    ).to(self.device)

                    # nn_model = ContextUnet(
                    #     in_channels=self.exp_conf.get('channels', 3),
                    #     n_feat=self.exp_conf.get('n_feat', 64),
                    #     n_classes=len(self.label_space_meta[d_name]),
                    #     norm_type="instance",
                    #     dropout_p=0.10,
                    #     down1_skip_scale=0.25,
                    #     out_skip_scale=0.0,
                    # )
            
                    # ddpm = DDPM(
                    #     nn_model=nn_model,
                    #     betas=(1e-4, 0.02),
                    #     n_T=1000,
                    #     device=self.device,
                    #     drop_prob=0.30,
                    #     target_lowres=16,
                    # ).to(self.device)

                    ddpm.load_state_dict(self.global_ddpm_states[d_name])
                    ddpm.eval()
                    ddpm_dict[d_name] = ddpm

            if not active_datasets:
                return 
            
            use_new_ent = self.exp_conf.get('use_new_entropy_method', False)
            entropy_ratio = self.exp_conf.get('entropy_ratio', 1.0) 
            cs_threshold = self.exp_conf.get('cs_threshold', 1.0) 

            mapping = None

            if self.args.label_mapping == 'image-bi':
                mapping = label_mapping(
                    get_images_func=get_gen_images, 
                    dataset_ids=active_datasets,    
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=entropy_ratio,
                    use_new_entropy_method=use_new_ent, 
                    logger=self.logger, 
                    args=self.args,
                    gen_dict=ddpm_dict
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'image-single':
                mapping = single_direction_label_mapping(
                    get_images_func=get_gen_images,  
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=entropy_ratio, 
                    use_new_entropy_method=use_new_ent,
                    logger=self.logger,
                    args=self.args,
                    gen_dict=ddpm_dict 
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'feature-bi':
                dataset_feats = self.get_all_clients_averaged_features()
                mapping = feature_bi_direction_label_mapping(
                    dataset_features_dict=dataset_feats,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=entropy_ratio, 
                    use_new_entropy_method=use_new_ent,
                    logger=self.logger,
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'image-cs':
                mapping = image_cosine_similarity_mapping(
                    get_images_func=get_gen_images,
                    dataset_ids=active_datasets,
                    label_space_meta=dataset_label_space_meta,
                    cs_threshold=cs_threshold,
                    logger=self.logger,
                    args=self.args,
                    gen_dict=ddpm_dict
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'class_name':
                self.class_name_label_mapping()
            elif self.args.label_mapping == 'independent':
                self.independent_label_mapping()
            elif self.args.label_mapping == 'identical':
                self.identical_label_mapping()

        if (self.glob_iter + 1) >= self.start_mapping_epoch: 
            self.train_global_inference_model()
            self.test_global_inference_model()
    
    def train_global_inference_model(self):
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

        self.model.to(self.device)
        self.model.train()

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

            # nn_model = ContextUnet(
            #     in_channels=self.exp_conf.get('channels', 3),
            #     n_feat=self.exp_conf.get('n_feat', 64),
            #     n_classes=len(self.label_space_meta[d_name]),
            #     norm_type="instance",
            #     dropout_p=0.10,
            #     down1_skip_scale=0.25,
            #     out_skip_scale=0.0,
            # )
    
            # gen = DDPM(
            #     nn_model=nn_model,
            #     betas=(1e-4, 0.02),
            #     n_T=1000,
            #     device=self.device,
            #     drop_prob=0.30,
            #     target_lowres=16,
            # ).to(self.device)

            gen.load_state_dict(state_dict)
            gen.eval()
            generators[d_name] = gen

        criterion = nn.CrossEntropyLoss()
        
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
                                          guide_w=self.exp_conf.get("ddpm_guide_w", 0.0),
                                          label=local_id)

                    # x_gen, _ = gen.sample(
                    #     n_sample=self.global_samples_per_class,
                    #     size=img_size,
                    #     device=self.device,
                    #     guide_w=0.0,
                    #     label=local_id,
                    #     init_noise_scale=0.75,
                    #     reverse_noise_scale=0.25,
                    # )
                    
                    # x_gen = torch.clamp(x_gen, -1.0, 1.0)

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

        for epoch in tqdm(range(self.global_model_epochs), colour="green", ncols=100):
            epoch_loss = 0
            for gen_imgs, y_batch in train_loader:
                gen_imgs, y_batch = gen_imgs.to(self.device), y_batch.to(self.device)
                
                self.global_model_optimizer.zero_grad()
                _, logits = self.model(gen_imgs)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.global_model_optimizer.step()
                
                epoch_loss += loss.item()
            
            self.logger.log(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")
        
    def distribute_model(self):
        for client in self.selected_clients:
            d_name = client.dataset_name
            if d_name in self.global_ddpm_states:
                client.ddpm.load_state_dict(self.global_ddpm_states[d_name])


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
        self.logger.log("Saving checkpoints ...")

        dataset_classifiers = {}
        for ls_id, model_dict in self.global_models.items():
            if 'classifier' in model_dict:
                dataset_classifiers[ls_id] = model_dict['classifier'].state_dict()

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

        server_save_path = os.path.join(self.logger.log_dir, 'server_'+fname)
        torch.save(checkpoint, server_save_path)
        self.logger.log(f"[Server] Checkpoint saved to {server_save_path}")

        gen_dir = os.path.join(self.logger.log_dir, 'global_gans')
        os.makedirs(gen_dir, exist_ok=True)
        
        for d_name in self.global_ddpm_states.keys():
            gan_checkpoint = {
                'generator': self.global_ddpm_states[d_name]
            }
            gan_save_path = os.path.join(gen_dir, f'{d_name}_DDPM.pth')
            torch.save(gan_checkpoint, gan_save_path)
            self.logger.log(f"[Server] Global DDPM for {d_name} saved to {gan_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")