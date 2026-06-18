import torch
import copy
from collections import OrderedDict, defaultdict
import os
import random
import csv
from tqdm import tqdm
import torch.nn as nn
from torch.optim import *

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from utils.nets import ResNet, BasicBlock, DCGANGenerator

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

            if (r + 1) % self.test_interval == 0:
                self.evaluate_private()
                #self.record_metric()

            self.aggregate()

            # if r+1 == 40:
            #     self.save_model() 

        # self.save_metric()

        self.save_model()
        plot_accuracy_curves(self.dataset_acc_history, self.logger.log_dir, self.args, self.global_rounds, self.dirichlet_alpha)


    def aggregate(self):
        groups = defaultdict(list)
        for client in self.selected_clients:
            d_name = client.dataset_name  
            #d_name = getattr(client, 'group_name', client.dataset_name)
            groups[d_name].append(client)

            if d_name not in self.label_space_meta:
                self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} datasets)...")

        if self.index_matching == 'gan':
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

        elif self.index_matching == 'ddpm':
            for d_name, group_clients in groups.items():
                if hasattr(group_clients[0], 'ddpm') and group_clients[0].ddpm is not None:
                    ddpm_msg_list = [
                        (client.num_samples, client.ddpm.state_dict())
                        for client in group_clients
                    ]
                    w_ddpm = self.avg_weights(ddpm_msg_list)
                    self.global_ddpm_states[d_name] = w_ddpm

        if (self.glob_iter + 1) == self.start_mapping_epoch: 
            if self.index_matching == 'class_name':
                self.class_name_label_mapping()
            elif self.index_matching == 'real_image':
                self.real_img_label_mapping()
            elif self.index_matching == 'gan':
                self.gan_label_mapping()
            elif self.index_matching == 'ddpm':
                self.ddpm_label_mapping()
            elif self.index_matching == 'independent':
                self.independent_label_mapping()
            elif self.index_matching == 'identical':
                self.identical_label_mapping()

        if (self.glob_iter + 1) >= self.start_mapping_epoch: 
            self.train_global_inference_model()
            self.test_global_inference_model()

        
    def distribute_model(self):
        for client in self.selected_clients:
            d_name = client.dataset_name
            #d_name = getattr(client, 'group_name', client.dataset_name)
            if d_name in self.global_gen_states and d_name in self.global_dis_states:
                client.generator.load_state_dict(self.global_gen_states[d_name])
                client.discriminator.load_state_dict(self.global_dis_states[d_name])


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

        criterion = nn.CrossEntropyLoss()

        y_list = []
        for d_name, mapping in self.local_id_to_global_id.items():
            if d_name not in generators:
                continue
            for local_id, global_id in mapping.items():
                y_list.extend([(d_name, local_id, global_id)] * self.global_samples_per_class)

        if not y_list:
            return
        
        for epoch in tqdm(range(self.global_model_epochs), colour="green", ncols=100):
            epoch_loss = 0
            random.shuffle(y_list)

            for i in range(0, len(y_list), self.batch_size):
                batch_info = y_list[i:i+self.batch_size]
                curr_batch_size = len(batch_info)
                
                gen_imgs = torch.zeros(curr_batch_size, 3, 32, 32).to(self.device)
                y_batch = torch.zeros(curr_batch_size, dtype=torch.long).to(self.device)

                d_to_local_ids = defaultdict(list)
                d_to_indices = defaultdict(list)
                
                for idx, (d_name, l_id, g_id) in enumerate(batch_info):
                    d_to_local_ids[d_name].append(l_id)
                    d_to_indices[d_name].append(idx)
                    
                for d_name, l_ids in d_to_local_ids.items():
                    gen = generators[d_name]
                    n_samples = len(l_ids)
                    z = torch.randn(n_samples, self.exp_conf.get('gen_noise_dim', 128)).to(self.device)
                    y_local = torch.tensor(l_ids, dtype=torch.long).to(self.device)
                    
                    with torch.no_grad():
                        imgs = gen(z, y_local)
                        
                    indices = d_to_indices[d_name]
                    gen_imgs[indices] = imgs
                    
                    g_ids = [batch_info[idx][2] for idx in indices]
                    y_batch[indices] = torch.tensor(g_ids, dtype=torch.long).to(self.device)
                
                self.global_model_optimizer.zero_grad()
                
                _, logits = self.model(gen_imgs)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.global_model_optimizer.step()
                
                epoch_loss += loss.item()
            
            self.logger.log(f"Epoch {epoch} Loss: {epoch_loss:.4f}")

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

        if self.model is not None:
            global_model_path = os.path.join(self.logger.log_dir, 'server_global_model.pth')
            torch.save(self.model.state_dict(), global_model_path)

        gan_dir = os.path.join(self.logger.log_dir, 'global_gans')
        os.makedirs(gan_dir, exist_ok=True)
        
        for d_name in self.global_gen_states.keys():
            gan_checkpoint = {
                'generator': self.global_gen_states[d_name],
                'discriminator': self.global_dis_states[d_name]
            }
            gan_save_path = os.path.join(gan_dir, f'{d_name}_GAN.pth')
            torch.save(gan_checkpoint, gan_save_path)
            self.logger.log(f"[Server] Global GAN for {d_name} saved to {gan_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")