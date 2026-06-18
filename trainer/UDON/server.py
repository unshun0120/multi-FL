import csv

import torch
import copy
from collections import defaultdict
import json
from torch import nn
from torch.nn import *
from torch.optim import *
import torch.nn.functional as F
from tqdm import tqdm
import random
import os
import numpy as np

from trainer.BaseFL.server import Server as BaseServer
from utils.nets import Classifier, ResNet, BasicBlock 
from utils.loss import RelationDistillationLoss
from utils.train_utils import evaluate_model
from data.datasets import get_readable_class_names

DATA_ROOT = './data/raw'

class Server(BaseServer):
    def __init__(self, **exp_conf):
        super(Server, self).__init__(**exp_conf)
        
        self.rd_loss_fn = RelationDistillationLoss()

        self.global_heads = None
        self.shared_classifier = None

    def aggregate(self):
        super().aggregate()

        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            self.train_global_inference_model()
            self.test_global_inference_model()

    def train_global_inference_model(self):
        print("Server: Training on Public Dataset...")
        
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
            self.global_heads = nn.ModuleDict()
            for d_name, mapping in self.local_id_to_global_id.items():
                self.global_heads[d_name] = nn.Linear(256, len(mapping), bias=False)

            self.shared_classifier = nn.Linear(256, num_global_classes, bias=False)
            self.model.classifier = self.shared_classifier 
            
            params = list(self.model.feature_extractor.parameters()) + \
                     list(self.model.adapter.parameters()) + \
                     list(self.global_heads.parameters()) + \
                     list(self.shared_classifier.parameters())
            
            self.global_model_optimizer = eval(self.global_model_optim_name)(params, self.global_model_optim_lr)

        self.model.train()
        self.model.to(self.device)
        self.global_heads.train()
        self.global_heads.to(self.device)
        self.shared_classifier.train()
        self.shared_classifier.to(self.device)

        dataset_meta = []
        domain_loss_history = {}
        total_batch_count = 0  

        client_groups = defaultdict(list)
        for client in self.selected_clients:
            client_groups[client.dataset_name].append(client)

        for dataset_name, client_list in client_groups.items():
            loader = self.train_loader[dataset_name]
            
            active_teachers = []
            for client in client_list:
                active_teachers.append({
                    'fe': client.model.feature_extractor.to(self.device).eval(),
                    'cls': client.model.classifier.to(self.device).eval(),
                    'adapter': client.model.adapter.to(self.device).eval() if hasattr(client.model, 'adapter') else None
                })
            
            mapping = self.local_id_to_global_id.get(dataset_name, {})
            num_classes = len(mapping)
            g_indices = torch.tensor([mapping[i] for i in range(num_classes)], dtype=torch.long).to(self.device)

            dataset_meta.append({
                'name': dataset_name,
                'teachers': active_teachers,
                'num_classes': num_classes,
                'g_indices': g_indices,
                'loader_len': len(loader),
            })

            domain_loss_history[dataset_name] = 1.0  
            samples_per_class = self.exp_conf.get('global_samples_per_class', 64)
            #total_batch_count += len(loader)
            total_batch_count += (samples_per_class * num_classes) // self.batch_size

        T = 2.0
        samples_per_class = self.exp_conf.get('global_samples_per_class', 64)
        batch_size = self.exp_conf.get('batch_size', 64)

        for epoch in tqdm(range(self.global_model_epochs), colour="green", ncols=100):
            self.logger.log(f"--- Epoch {epoch+1}/{self.global_model_epochs} ---")
            
            #dataset_iterators = {d['name']: iter(self.train_loader[d['name']]) for d in dataset_meta}
            
        #     epoch_batches = defaultdict(list)
        #     for d_info in dataset_meta:
        #         d_name = d_info['name']
        #         mapping = self.local_id_to_global_id.get(d_name, {})
        #         label_count = defaultdict(int)
                
        #         current_epoch_images = []
        #         current_epoch_labels = []
                
        #         for x, y in self.train_loader[d_name]:
        #             for i in range(len(y)):
        #                 lbl = int(y[i])
        #                 if lbl in mapping and label_count[lbl] < samples_per_class:
        #                     label_count[lbl] += 1
        #                     current_epoch_images.append(x[i])
        #                     current_epoch_labels.append(y[i]) 

        #                 if len(current_epoch_images) == self.batch_size:
        #                     epoch_batches[d_name].append((torch.stack(current_epoch_images), torch.tensor(current_epoch_labels)))
        #                     current_epoch_images = []
        #                     current_epoch_labels = []
                    
        #             if all(label_count[l] >= samples_per_class for l in mapping.keys()):
        #                 break
                        
        #         if len(current_epoch_images) > 0:
        #             epoch_batches[d_name].append((torch.stack(current_epoch_images), torch.tensor(current_epoch_labels)))

        #     dataset_iterators = {d_name: iter(batches) for d_name, batches in epoch_batches.items()}
                
        #     total_loss = 0.0
        #     steps = 0
            
        #     for step in range(total_batch_count):
        #         dataset_names = list(domain_loss_history.keys())
        #         losses = torch.tensor([domain_loss_history[name] for name in dataset_names])
                
        #         if epoch == 0:
        #             probs = torch.ones_like(losses) / len(losses)
        #         else:
        #             probs = losses / losses.sum()

        #         sampled_idx = torch.multinomial(probs, 1).item()
        #         selected_dataset_name = dataset_names[sampled_idx]
                
        #         d_config = next(d for d in dataset_meta if d['name'] == selected_dataset_name)
        #         g_indices = d_config['g_indices']

        #         current_mapping = self.local_id_to_global_id.get(selected_dataset_name, {})
                
        #         try:
        #             images, labels = next(dataset_iterators[selected_dataset_name])
        #         except StopIteration:
        #             dataset_iterators[selected_dataset_name] = iter(epoch_batches[selected_dataset_name])
        #             images, labels = next(dataset_iterators[selected_dataset_name])

        #         images = images.to(self.device)
        #         num_classes = d_config['num_classes']
        #         global_labels = torch.tensor([current_mapping[l.item()] for l in labels], dtype=torch.long).to(self.device)

        #         t_logits_sum = 0.0
        #         t_feat_sum = 0.0
        #         for teacher in d_config['teachers']:
        #             with torch.no_grad():
        #                 feat = teacher['fe'](images)
        #                 feat = torch.flatten(feat, 1)
        #                 if teacher['adapter']: feat = teacher['adapter'](feat)
        #                 logits = teacher['cls'](feat)
        #                 if logits.shape[1] > num_classes:
        #                     logits = logits[:, :num_classes]
                        
        #                 t_feat_sum += feat
        #                 t_logits_sum += logits
                        
        #         t_feat_avg = t_feat_sum / len(d_config['teachers'])
        #         t_logits_avg = t_logits_sum / len(d_config['teachers'])
                    
        #         s_feat, s_logits = self.model(images)

        #         s_logits_subset = self.global_heads[selected_dataset_name](s_feat)

        #         s_log_probs = F.log_softmax(s_logits_subset / T, dim=1)
        #         t_probs = F.softmax(t_logits_avg / T, dim=1)
        #         loss_kd = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (T * T)
        #         loss_rd = self.rd_loss_fn(s_feat, t_feat_avg)

        #         s_universal_logits = self.shared_classifier(s_feat)
        #         s_univ_logits_subset = s_universal_logits[:, g_indices]
        #         s_univ_log_probs = F.log_softmax(s_univ_logits_subset / T, dim=1)
        #         loss_universal_kd = F.kl_div(s_univ_log_probs, t_probs, reduction='batchmean') * (T * T)
                
        #         loss_ce = F.cross_entropy(s_universal_logits, global_labels)

        #         loss = 1.0 * loss_kd + 1.0 * loss_universal_kd + 0.1 * loss_rd + 1.0 * loss_ce
                
        #         self.global_model_optimizer.zero_grad()
        #         loss.backward()
        #         self.global_model_optimizer.step()
                
        #         total_loss += loss.item()

        #         current_loss_val = loss.item()
        #         domain_loss_history[selected_dataset_name] = max(current_loss_val, 1e-6)
            
        #     self.logger.log(f"Epoch {epoch+1} done. Avg Loss: {total_loss / total_batch_count:.4f}")

        # self.model.to('cpu')
        # self.global_heads.to('cpu')
        # self.shared_classifier.to('cpu')
        # print("[Server] Dataset-Specific Distillation Finished.")

            all_data = [] # 儲存格式: (影像, 原始 local label, dataset_name)
            for d_info in dataset_meta:
                d_name = d_info['name']
                mapping = self.local_id_to_global_id.get(d_name, {})
                label_count = defaultdict(int)
                
                for x, y in self.train_loader[d_name]:
                    for i in range(len(y)):
                        lbl = int(y[i])
                        if lbl in mapping and label_count[lbl] < samples_per_class:
                            label_count[lbl] += 1
                            all_data.append((x[i], lbl, d_name))
                            
                    if all(label_count[l] >= samples_per_class for l in mapping.keys()):
                        break
            
            random.shuffle(all_data)
            
            total_loss = 0.0
            steps = 0
            
            for i in range(0, len(all_data), batch_size):
                batch_data = all_data[i:i+batch_size]
                
                batch_by_domain = defaultdict(list)
                for img, lbl, d_name in batch_data:
                    batch_by_domain[d_name].append((img, lbl))
                
                self.global_model_optimizer.zero_grad()
                batch_total_loss = 0.0
                
                for d_name, items in batch_by_domain.items():
                    sub_images, sub_labels = zip(*items)
                    sub_images = torch.stack(sub_images).to(self.device)
                    sub_labels = torch.tensor(sub_labels, dtype=torch.long)
                    
                    d_config = next(d for d in dataset_meta if d['name'] == d_name)
                    g_indices = d_config['g_indices']
                    current_mapping = self.local_id_to_global_id.get(d_name, {})
                    global_labels = torch.tensor([current_mapping[l.item()] for l in sub_labels], dtype=torch.long).to(self.device)
                    num_classes = d_config['num_classes']
                    
                    t_logits_sum = 0.0
                    t_feat_sum = 0.0
                    for teacher in d_config['teachers']:
                        with torch.no_grad():
                            feat = teacher['fe'](sub_images)
                            feat = torch.flatten(feat, 1)
                            if teacher['adapter']: feat = teacher['adapter'](feat)
                            logits = teacher['cls'](feat)
                            if logits.shape[1] > num_classes:
                                logits = logits[:, :num_classes]
                            t_feat_sum += feat
                            t_logits_sum += logits
                            
                    t_feat_avg = t_feat_sum / len(d_config['teachers'])
                    t_logits_avg = t_logits_sum / len(d_config['teachers'])
                        
                    s_feat, s_logits = self.model(sub_images)
                    s_logits_subset = self.global_heads[d_name](s_feat)

                    s_log_probs = F.log_softmax(s_logits_subset / T, dim=1)
                    t_probs = F.softmax(t_logits_avg / T, dim=1)
                    loss_kd = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (T * T)
                    loss_rd = self.rd_loss_fn(s_feat, t_feat_avg)

                    s_universal_logits = self.shared_classifier(s_feat)
                    s_univ_logits_subset = s_universal_logits[:, g_indices]
                    s_univ_log_probs = F.log_softmax(s_univ_logits_subset / T, dim=1)
                    loss_universal_kd = F.kl_div(s_univ_log_probs, t_probs, reduction='batchmean') * (T * T)
                    loss_ce = F.cross_entropy(s_universal_logits, global_labels)

                    sub_loss = 1.0 * loss_kd + 1.0 * loss_universal_kd + 0.1 * loss_rd + 1.0 * loss_ce
                    weight = len(sub_images) / len(batch_data)
                    batch_total_loss += sub_loss * weight
                
                batch_total_loss.backward()
                self.global_model_optimizer.step()
                
                total_loss += batch_total_loss.item()
                steps += 1
                
            self.logger.log(f"Epoch {epoch+1} done. Avg Loss: {total_loss / max(1, steps):.4f}")

        self.model.to('cpu')
        self.global_heads.to('cpu')
        self.shared_classifier.to('cpu')
        print("[Server] Dataset-Specific Distillation Finished.")

    def distribute_model(self):
        pass


    def save_model(self, fname='checkpoints.pth'):
        self.logger.log("Saving UDON Server Checkpoints ...")

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
            # torch.save({
            #     'feature_extractor': self.model.feature_extractor.state_dict(),
            #     'adapter': self.model.adapter.state_dict()
            # }, global_model_path)

            full_state_dict = self.model.state_dict()
            
            if self.shared_classifier is not None:
                full_state_dict['classifier.weight'] = self.shared_classifier.weight
                
                if getattr(self.shared_classifier, 'bias', None) is not None:
                    full_state_dict['classifier.bias'] = self.shared_classifier.bias
                elif 'classifier.bias' in full_state_dict:
                    full_state_dict['classifier.bias'] = torch.zeros_like(full_state_dict['classifier.bias'])
            
            torch.save(full_state_dict, global_model_path)

            for d_name, head in self.global_heads.items():
                domain_head_path = os.path.join(self.logger.log_dir, f'server_{d_name}_global_head.pth')
                torch.save(head.state_dict(), domain_head_path)
            
            if hasattr(self, 'shared_classifier') and self.shared_classifier is not None:
                shared_classifier_path = os.path.join(self.logger.log_dir, 'server_shared_classifier.pth')
                torch.save(self.shared_classifier.state_dict(), shared_classifier_path)

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)
        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All clients saved in {clients_dir}/")