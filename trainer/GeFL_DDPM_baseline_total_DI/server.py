"""
Server for GeFL (DeepInversion / Data-Free KD)
"""

import torch
import copy
from collections import OrderedDict, defaultdict, deque
import os
import csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from utils.nets import ResNet, BasicBlock, DCGANGenerator

from label_mapping.label_mapping_utils import (
    label_mapping, evaluate_mapping_results, 
    feature_bi_direction_label_mapping, single_direction_label_mapping,
    get_gen_images, global_to_local_mapping, clear_image_caches,
    image_cosine_similarity_mapping,
)
from utils.nets import ResNet, BasicBlock

from utils.loss import Gen_DiversityLoss
from utils.nets import ConditionalImageGenerator, NLGenerator
from utils.DFKD_utils import (
    KLDiv, JSDiv, evaluate_student_model,
    jitter_and_flip, get_image_prior_losses, DeepInversionHook,
    reptile_grad, fomaml_grad, get_fast_augmentation, 
    get_nayer_label_embedding, nayer_cross_entropy,
)

GEN_CONFIG = {
    'img_num_samples': 64,
    'feat_gen_noise_dim': 128,
    'student_lr': 1e-3,
    'kd_steps': 1,     
}

di_weight = {
    'epochs': 20, 
    'g_steps': 1500, 
    'lr': 0.02,
    'ce': 1.0,      
    'bn': 0.05,      
    'tv': 0.005,    
    'l2': 0.0,    
    'adv': 0.0, 
    'student_lr': 0.2,
}

class Server(BaseServer):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.global_gen_states = {}
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

            self.aggregate()

        self.save_model()

    def aggregate(self):
        groups = defaultdict(list)
        for client in self.selected_clients:
            d_name = client.dataset_name  
            groups[d_name].append(client)

            if d_name not in self.label_space_meta:
                self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} datasets)...")

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

        if (self.glob_iter + 1) == self.start_mapping_epoch: 

            clear_image_caches()

            if self.args.label_mapping == 'class_name':
                self.class_name_label_mapping()
            elif self.args.label_mapping == 'independent':
                self.independent_label_mapping()
            elif self.args.label_mapping == 'identical':
                self.identical_label_mapping()

        if (self.glob_iter + 1) >= self.start_mapping_epoch: 
            self.global_gen_states = {}

            image_pools_dict = self.train_generators_DeepInversion(
                clients_dict=dataset_clients_dict,
                label_space_meta=dataset_label_space_meta,
                dataset_meta={}, 
                device=self.device,
                logger=self.logger,
                use_new_gen_method=self.exp_conf.get('use_new_gen_method', True)
            )

            for d_name, pool in image_pools_dict.items():
                self.global_gen_states[d_name] = pool

                
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

        criterion = nn.CrossEntropyLoss()
        noise_dim = self.exp_conf.get('gen_noise_dim', 128)
        dataset_x = []
        dataset_y = []
        
        for d_name, mapping in self.local_id_to_global_id.items():
            if d_name not in self.global_gen_states: continue
            
            pool = self.global_gen_states[d_name]

            for local_id, global_id in mapping.items():
                if local_id in pool and len(pool[local_id]) > 0:
                    imgs = pool[local_id].to(self.device)
                    
                    indices = torch.randint(0, len(imgs), (self.global_samples_per_class,))
                    x_gen = imgs[indices]

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
        total_samples = sum([w[0] for w in weights_list])
        avg_params = OrderedDict()
        
        for name in weights_list[0][1].keys():
            avg_params[name] = torch.zeros_like(weights_list[0][1][name], dtype=torch.float32)
            
            for num_samples, params in weights_list:
                avg_params[name] += params[name] * (num_samples / total_samples)
                
        return avg_params


    def train_generators_DeepInversion(self, clients_dict, label_space_meta, dataset_meta, device, logger, 
                                    use_new_gen_method=True, client_label_mask_dict=None, student_model=None, test_loaders_dict=None):
        
        method_name = "New Gen Method" if use_new_gen_method else "Old Gen Method"
        logger.log(f"[Testing] Training Per-Dataset Generators Offline ({method_name})...")
        logger.log(f"--- DeepInversion Generator Configuration ---")
        for k, v in di_weight.items():
            logger.log(f"    {k}: {v}")
        logger.log(f"------------------------------------")

        generators_dict = {}

        for ls_id, dataset_clients in clients_dict.items():
            if len(dataset_clients) == 0:
                continue

            current_student = student_model[ls_id] if (student_model is not None and ls_id in student_model) else None
            if current_student is not None:
                student_optimizer = torch.optim.SGD(current_student.parameters(), lr=di_weight['student_lr'], momentum=0.9, weight_decay=1e-4)
        
            class_names = label_space_meta[ls_id]
            num_local_classes = len(class_names)
            dataset_clients.sort(key=lambda m: getattr(m, 'client_id', 0)) 
            
            if use_new_gen_method and client_label_mask_dict is not None and ls_id in client_label_mask_dict:
                raw_mask = client_label_mask_dict[ls_id]
                current_mask = raw_mask[:len(dataset_clients)].to(device)
                valid_client_indices = [i for i in range(len(dataset_clients)) if current_mask[i].sum() > 0]
            else:
                current_mask = torch.ones((len(dataset_clients), num_local_classes), device=device)
                valid_client_indices = list(range(len(dataset_clients)))

            batch_size = 64
            epoch_loss_tracker = []
            
            all_client_bn_hooks = []
            for client_model in dataset_clients:
                client_model.eval() 
                for param in client_model.parameters():      
                    param.requires_grad = False

                hooks = []
                for module in client_model.modules():
                    if hasattr(module, 'inplace'):
                        module.inplace = False
                    if isinstance(module, torch.nn.BatchNorm2d):
                        hooks.append(DeepInversionHook(module))
                all_client_bn_hooks.append(hooks)

            valid_client_count = len(valid_client_indices)
            if valid_client_count > 0:
                all_valid_labels = torch.where(current_mask[valid_client_indices].sum(dim=0) > 0)[0]
            else:
                all_valid_labels = torch.arange(num_local_classes, device=device)

            class_image_pools = {c: [] for c in range(num_local_classes)}
            image_pool = deque(maxlen=2000)
            student_acc_history = [] 

            for epoch in tqdm(range(di_weight['epochs']), colour='blue', ncols=100, desc=f"Gen:{ls_id}"):
                if current_student is not None:
                    current_student.eval()

                batch_labels = all_valid_labels[torch.randint(0, len(all_valid_labels), (batch_size,)).to(device)]
                inputs = torch.randn(batch_size, 3, 32, 32, device=device).requires_grad_()
                pixel_optimizer = torch.optim.Adam([inputs], lr=di_weight['lr'], betas=[0.5, 0.99])

                best_cost = 1e6
                best_inputs = None

                for it in range(di_weight['g_steps']):
                    pixel_optimizer.zero_grad() 
                    
                    batch_loss = 0.0
                    clients_contributed = 0 

                    for client_idx in valid_client_indices:
                        client_model = dataset_clients[client_idx]
                        client_mask = current_mask[client_idx] 
                        known_mask = client_mask[batch_labels].bool() 
                        
                        if not known_mask.any(): continue 
                        
                        sub_imgs = inputs[known_mask]
                        sub_labels = batch_labels[known_mask]
                        
                        preds = client_model(sub_imgs)
                        if isinstance(preds, tuple): preds = preds[1] # 處理 Tuple 回傳
                        logits_t = preds

                        cls_loss = F.cross_entropy(logits_t, sub_labels)
                        
                        bn_loss = 0.0
                        client_hooks = all_client_bn_hooks[client_idx]
                        if len(client_hooks) > 0 and di_weight['bn'] != 0:
                            bn_loss = sum([h.r_feature for h in client_hooks]) / len(client_hooks)
                        
                        sub_imgs_prior = inputs[known_mask]
                        tv_loss = get_image_prior_losses(sub_imgs_prior)
                        l2_loss = torch.norm(sub_imgs_prior, 2)

                        client_loss = (di_weight['ce'] * cls_loss) + \
                                        (di_weight['bn'] * bn_loss) + \
                                        (di_weight['tv'] * tv_loss) + \
                                        (di_weight['l2'] * l2_loss)
                        
                        if current_student is not None and di_weight['adv'] > 0:
                            _, logits_s = current_student(sub_imgs)
                            loss_adv = -JSDiv(logits_s, logits_t.detach(), T=3.0)
                            client_loss += di_weight['adv'] * loss_adv
                        
                        batch_loss = batch_loss + client_loss
                        clients_contributed += 1

                    if clients_contributed > 0:
                        avg_loss = batch_loss / clients_contributed
                        avg_loss.backward()
                        pixel_optimizer.step()
                        
                        if avg_loss.item() < best_cost:
                            best_cost = avg_loss.item()
                            best_inputs = inputs.detach().clone()

                if best_inputs is not None:
                    epoch_loss_tracker.append(best_cost)
                    for i in range(batch_size):
                        lbl = batch_labels[i].item()
                        class_image_pools[lbl].append(best_inputs[i].detach().cpu())
                    for client_idx in valid_client_indices:
                        client_mask = current_mask[client_idx] 
                        known_mask = client_mask[batch_labels].bool() 
                        if known_mask.any():
                            sub_imgs = best_inputs[known_mask]
                            image_pool.append((client_idx, sub_imgs.detach()))

            if len(epoch_loss_tracker) > 0:
                avg_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
                logger.log(f"    Dataset [{ls_id}] Generator | Epochs: {di_weight['epochs']} | Final Loss: {avg_loss:.4f}")

            for hooks in all_client_bn_hooks:
                for h in hooks:
                    if hasattr(h, 'close'): h.close()
                    elif hasattr(h, 'remove'): h.remove()

            for client_model in dataset_clients:
                for param in client_model.parameters():
                    param.requires_grad = True

            for c in range(num_local_classes):
                if len(class_image_pools[c]) > 0:
                    class_image_pools[c] = torch.stack(class_image_pools[c])
                else:
                    class_image_pools[c] = torch.full((GEN_CONFIG['img_num_samples'], 3, 32, 32), -1.0)

            generators_dict[ls_id] = class_image_pools
        return generators_dict

    def get_pool_images(self, pool_dict, local_id, num_samples, device, **kwargs):
        if local_id in pool_dict and len(pool_dict[local_id]) > 0:
            pool = pool_dict[local_id]
            indices = torch.randint(0, len(pool), (num_samples,))
            return pool[indices].to(device)


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
            'global_registry': getattr(self, 'local_id_to_global_id', {}),
            'label_space_meta': self.label_space_meta,
            'global_feature_dim': getattr(self, 'global_feature_dim', 256),
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
        
        for d_name in self.global_gen_states.keys():
            gan_checkpoint = {
                'generator': self.global_gen_states[d_name]
            }
            gan_save_path = os.path.join(gen_dir, f'{d_name}_GAN.pth')
            torch.save(gan_checkpoint, gan_save_path)
            self.logger.log(f"[Server] Inversion GAN for {d_name} saved to {gan_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")