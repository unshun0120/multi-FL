"""
Server for GeFL DDPM baseline
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
from contextlib import redirect_stdout

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from utils.nets import ResNet, BasicBlock, DCGANGenerator

from label_mapping.label_mapping_utils import (
    label_mapping, evaluate_mapping_results, 
    feature_bi_direction_label_mapping, single_direction_label_mapping,
    get_gen_images, global_to_local_mapping, clear_image_caches,
    image_cosine_similarity_mapping, missing_link_label_mapping
)
from label_mapping.slam_dunk import slam_dunk_label_mapping
from utils.nets import ResNet, BasicBlock

class Server(BaseServer):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.global_gen_states = {}
        self.global_dis_states = {}
        self.model = None

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

        self.save_model()
        # plot_accuracy_curves(self.dataset_acc_history, self.logger.log_dir, self.args, self.global_rounds, self.dirichlet_alpha)


    def get_all_clients_averaged_features(self):
        dataset_features = {}
        for client in self.clients:
            d_id = f"client_{client.id}"
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
            gen_msg_list = [
                (client.num_samples, client.generator.state_dict())
                for client in group_clients
            ]
            w_gen = self.aggregate_weights(gen_msg_list)
            self.global_gen_states[d_name] = w_gen

            dis_msg_list = [
                (client.num_samples, client.discriminator.state_dict())
                for client in group_clients
            ]
            w_dis = self.aggregate_weights(dis_msg_list)
            self.global_dis_states[d_name] = w_dis

        # if (self.glob_iter + 1) == self.start_mapping_epoch: 
        if (self.glob_iter + 1) % 5 == 0:
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

            generators = {}
            for d_name, state_dict in self.global_gen_states.items():
                num_local_classes = len(self.label_space_meta.get(d_name, []))
                if num_local_classes == 0: continue
                
                gen = DCGANGenerator(
                    num_classes=num_local_classes,
                    noise_dim=self.exp_conf.get('gen_noise_dim', 128),
                    img_size=self.exp_conf.get('img_size', 32),
                    channels=self.exp_conf.get('channels', 3)
                ).to(self.device)
                gen.load_state_dict(state_dict)
                gen.eval()
                generators[d_name] = gen

            if not active_datasets:
                return 
            
            use_new_ent = self.exp_conf.get('use_new_entropy_method', False)
            #entropy_ratio = self.exp_conf.get('entropy_ratio', 1.0) 
            entropy_ratios = [i / 10 for i in range(1, 11)]
            missing_threshold = [i / 10 for i in range(1, 11)]
            cs_threshold = self.exp_conf.get('cs_threshold', 1.0) 

            csv_filename = os.path.join(self.logger.log_dir, f'{self.args.label_mapping}_noniid_mapping_acc.csv')

            mapping_log_dir = os.path.join(self.logger.log_dir, 'mapping_summary')
            os.makedirs(mapping_log_dir, exist_ok=True)
            mapping_log_filename = os.path.join(mapping_log_dir, f'global_round_{self.glob_iter + 1}.log')

            for threshold in missing_threshold:
                mapping = missing_link_label_mapping(
                    get_images_func=get_gen_images,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    missing_threshold=threshold,
                    logger=self.logger, 
                    args=self.args,
                    gen_dict=generators
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

                with open(mapping_log_filename, 'a', encoding='utf-8') as f:
                    f.write("\n" + "#" * 50 + "\n")
                    f.write(f"Missing Linke Threshold = {threshold}\n")
                    f.write("#" * 50 + "\n")

                    with redirect_stdout(f):
                        global_to_local_mapping(
                            mapping,
                            logger=None,
                            label_space_meta=dataset_label_space_meta
                        )

                metrics = evaluate_mapping_results(
                    dataset_ids=active_datasets,
                    label_space_meta=dataset_label_space_meta,
                    local_id_to_global_id=mapping,
                )

                file_exists = os.path.isfile(csv_filename)

                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    if not file_exists:
                        writer.writerow(['global_round', 'entropy_ratio', 
                                        'recall', 'specificity', 'precision', 
                                        'average_accuracy', 'f1_score', 'mcc',
                                        'TP', 'FP', 'TN', 'FN'])

                    writer.writerow([(self.glob_iter + 1), threshold,
                                    metrics['Recall'], metrics['Specificity'], metrics['Precision'], 
                                    metrics['AvgAccuracy'], metrics['F1-Score'], metrics['MCC'],
                                    metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']])

        # if (self.glob_iter + 1) >= self.start_mapping_epoch: 
        #     self.train_global_inference_model()
        #     self.test_global_inference_model()
    
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
        for client in self.clients:
            d_name = f"client_{client.id}"
            client.generator.to(self.device)
            client.generator.eval()
            generators[d_name] = client.generator

        criterion = nn.CrossEntropyLoss()
        
        img_size = (self.exp_conf.get('channels', 3), self.exp_conf.get('img_size', 32), self.exp_conf.get('img_size', 32))
        
        dataset_x = []
        dataset_y = []
        noise_dim = self.exp_conf.get('gen_noise_dim', 128)
        
        for d_name, mapping in self.local_id_to_global_id.items():
            if d_name not in generators:
                continue
            gen = generators[d_name]
            gen.eval()

            for local_id, global_id in mapping.items():
                z = torch.randn(self.global_samples_per_class, noise_dim).to(self.device)
                y_local = torch.full((self.global_samples_per_class,), local_id, dtype=torch.long).to(self.device)
                with torch.no_grad():
                    x_gen = gen(z, y_local)

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
        pass


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
        
        for client in self.clients:
            d_name = f"client_{client.id}"
            gan_checkpoint = {
                'generator': client.generator.state_dict()
            }
            gan_save_path = os.path.join(gen_dir, f'{d_name}_GAN.pth')
            torch.save(gan_checkpoint, gan_save_path)
            self.logger.log(f"[Server] Local GAN for {d_name} saved to {gan_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")