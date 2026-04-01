import copy
import os
import numpy as np
import torch
from torch.nn import *
from torch.optim import *
import torch.nn.functional as F
from collections import defaultdict
import random
from tqdm import tqdm
import torch.nn as nn

from trainer.BaseFL.server import Server as Base_Server
from utils.nets import ConditionalGenerator, Classifier, ConditionalImageGenerator

class Server(Base_Server):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.global_gen_epochs = kwargs.get('global_gen_epochs', 10) 
        self.gen_noise_dim = kwargs.get('gen_noise_dim', 128) 
        self.gen_optim_lr = kwargs.get('gen_optim_lr', 1e-3)
        self.gen_optim_name =  kwargs.get('gen_optim', 'Adam')
        self.global_gen = None
        self.gen_optimizer = None

        self.gen_observer_weight = kwargs.get('gen_observer_weight', 1.0) 

        self.rebuild_generic_model_epochs = kwargs.get('rebuild_generic_model_epochs', 15)


    def aggregate(self):
        super().aggregate()

        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            self.train_generator()
            self.train_global_inference_model()


    def train_generator(self):
        all_global_ids = set()
        for d_name, mapping in self.local_id_to_global_id.items():
            for l_id, g_id in mapping.items():
                all_global_ids.add(g_id)
        all_global_ids = list(all_global_ids)

        global_num_classes = max(all_global_ids) + 1
        if self.global_gen is None:
            self.global_gen = ConditionalImageGenerator(
                num_classes=global_num_classes, 
                noise_dim=self.gen_noise_dim,
                img_channels=3,
                img_size=32   
            ).to(self.device)
            self.gen_optimizer = eval(self.gen_optim_name)(self.global_gen.parameters(), self.gen_optim_lr)

        for client in self.selected_clients:
            client.model.to(self.device)
            client.model.eval()
            for param in client.model.parameters():
                param.requires_grad = False

        self.global_gen.train()

        self.logger.log("[Server] Training Image-Based Global Generator...")
        for epoch in tqdm(range(self.global_gen_epochs), colour="blue", ncols=100):
            epoch_loss = 0
            random.shuffle(all_global_ids)
            
            for i in range(0, len(all_global_ids), self.batch_size):
                batch_ids = all_global_ids[i : i + self.batch_size]
                curr_batch = len(batch_ids)

                labels_input = torch.tensor(batch_ids).to(self.device)
                z = torch.randn(curr_batch, self.gen_noise_dim).to(self.device)
                gen_imgs = self.global_gen(z, labels_input)
                
                count_expert = 0
                count_observer = 0
                batch_loss_expert = 0
                batch_loss_observer = 0
                
                self.gen_optimizer.zero_grad()

                for client in self.selected_clients:
                    d_name = client.dataset_name
                    client.model.eval() 

                    _, logits = client.model(gen_imgs)

                    global_to_local = {}
                    if d_name in self.local_id_to_global_id:
                        for l_id, g_id in self.local_id_to_global_id[d_name].items():
                            global_to_local[g_id] = l_id

                    # 看這個 client 認不認識 batch 裡的 global_id
                    target_list = []
                    for gid in batch_ids:
                        if gid in global_to_local:
                            # 認識 -> 存入他原生資料集的 local id
                            target_list.append(global_to_local[gid])
                        else:
                            # 不認識 -> 存 -1
                            target_list.append(-1)

                    target_tensor = torch.tensor(target_list).to(self.device)
                    # boolean tensor
                    mask_expert = (target_tensor != -1)

                    # expert (classification loss) 
                    if mask_expert.any():
                        # 取mask=true的算loss
                        loss_ce = F.cross_entropy(logits[mask_expert], target_tensor[mask_expert])
                        batch_loss_expert += loss_ce
                        count_expert += 1

                    # observer (logit distillation)
                    if (~mask_expert).any():
                        ood_logits = logits[~mask_expert]
                        ood_probs = F.log_softmax(ood_logits, dim=1)
        
                        target_dir_soft_label = self.get_dir_soft_label(
                            num_classes=ood_logits.size(1),
                            batch_size=ood_logits.size(0)
                        )

                        loss_kl = F.kl_div(ood_probs, target_dir_soft_label, reduction='batchmean')
                        batch_loss_observer += loss_kl
                        count_observer += 1

                # normalization
                if count_expert > 0:
                    batch_loss_expert /= count_expert
                if count_observer > 0:
                    batch_loss_observer /= count_observer
                
                total_loss = batch_loss_expert + (self.gen_observer_weight * batch_loss_observer)

                total_loss.backward()
                self.gen_optimizer.step()
                
                epoch_loss += total_loss.item()

            self.logger.log(f"  Epoch {epoch+1}/{self.global_gen_epochs} | Loss: {epoch_loss:.4f}", print_to_console=False)

        self.logger.log("[Server] Image-Based Global Generator training finished.")

        for client in self.selected_clients:
            client.model.to('cpu') # Move back to CPU to save GPU memory
            for param in client.model.parameters():
                param.requires_grad = True


    def train_global_inference_model(self):
        self.logger.log("[Server] Training Final Global Inference Model ...")
        
        all_global_ids = set()
        for d_name, mapping in self.local_id_to_global_id.items():
            for l_id, g_id in mapping.items():
                all_global_ids.add(g_id)
        
        if len(all_global_ids) == 0:
            self.logger.log("Warning: No mappings found via local_id_to_global_id. Skipping global model training.")
            return

        num_global_classes = len(all_global_ids)
        self.logger.log(f"Detected {num_global_classes} global classes: {sorted(list(all_global_ids))}")

        # Adjust global model classifier output dimension
        if self.model.classifier.out_features != num_global_classes:
            self.logger.log(f"Adjusting Global Model head from {self.model.classifier.out_features} to {num_global_classes}")
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_global_classes)
        
        self.model.to(self.device)
        self.model.train()
        self.global_gen.eval() 

        batch_size = self.batch_size
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        all_global_ids = list(all_global_ids)

        for epoch in tqdm(range(self.rebuild_generic_model_epochs), colour="green", ncols=100):
            epoch_loss = 0
            random.shuffle(all_global_ids)
            
            for i in range(0, len(all_global_ids), batch_size):
                batch_ids = all_global_ids[i : i + batch_size]
                curr_batch_len = len(batch_ids)

                labels = torch.tensor(batch_ids).to(self.device)
                
                z = torch.randn(curr_batch_len, self.gen_noise_dim).to(self.device)
                with torch.no_grad():
                    gen_imgs = self.global_gen(z, labels)
                
                optimizer.zero_grad()
                
                _, logits = self.model(gen_imgs)
                
                targets = torch.tensor(batch_ids, dtype=torch.long).to(self.device)

                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            self.logger.log(f"Epoch {epoch} Loss: {epoch_loss:.4f}")


    def distribute_model(self):
        # gen_state_dict = None
        # if self.global_feature_gen is not None:
        #     gen_state_dict = self.global_feature_gen.state_dict()

        # for client in self.selected_clients:
        #     ls_id = str(client.class_name_set)
            
        #     if ls_id not in self.global_models:
        #         continue

        #     global_part = self.global_models[ls_id]

        #     if 'classifier' in global_part:
        #         client.model.classifier.load_state_dict(global_part['classifier'].state_dict())

        #     if gen_state_dict is not None:
        #         if hasattr(client, 'global_feature_gen') and client.global_feature_gen is not None:
        #             client.global_feature_gen.load_state_dict(gen_state_dict)

        pass
    

    def get_dir_soft_label(self, num_classes, batch_size):
        z = torch.randn(batch_size, self.gen_noise_dim, device=self.device)
        z = z.abs()
        z = torch.clamp(z, min=1.0)

        dir_alpha_niose = z.mean(dim=1, keepdim=True)
        dirichlet_alpha = dir_alpha_niose.expand(batch_size, num_classes)

        dist = torch.distributions.Dirichlet(dirichlet_alpha)
        dir_soft_labels = dist.sample()
        #print(dir_soft_labels)
        return dir_soft_labels
    
    def save_model(self, fname='checkpoints.pth'):
        dataset_classifiers = {}
        for ls_id, model_dict in self.global_models.items():
            if 'classifier' in model_dict:
                dataset_classifiers[ls_id] = model_dict['classifier'].state_dict()

        clients_state = {}
        for client in self.clients:
            clients_state[client.id] = client.model.state_dict()

        checkpoint = {
            'generator': self.global_gen.state_dict(),
            'clients': clients_state,
            'global_registry': self.local_id_to_global_id,
            'label_space_meta': self.label_space_meta,
            'global_feature_dim': self.global_feature_dim,
            'gen_noise_dim': self.gen_noise_dim,
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

        global_model_save_path = os.path.join(self.logger.log_dir, 'global_inference_model.pth')
        torch.save(self.model.state_dict(), global_model_save_path)
        self.logger.log(f"[Server] Global Inference Model saved to {global_model_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, 'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")



