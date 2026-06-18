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
from utils.nets import ConditionalGenerator, Classifier, ConditionalImageGenerator, ResNet, BasicBlock 
from utils.loss import Gen_DiversityLoss

class Server(Base_Server):
    def __init__(self, **exp_conf):
        super(Server, self).__init__(**exp_conf)

        self.global_gen_epochs = exp_conf.get('global_gen_epochs', 10) 
        self.gen_noise_dim = exp_conf.get('gen_noise_dim', 128) 
        self.gen_optim_lr = exp_conf.get('gen_optim_lr', 1e-3)
        self.gen_optim_name =  exp_conf.get('gen_optim', 'Adam')
        self.global_gen = None
        self.gen_optimizer = None

        self.gen_observer_weight = exp_conf.get('gen_observer_weight', 1.0) 
        self.global_samples_per_class = getattr(self.exp_conf, 'global_samples_per_class', 1)

    def distribute_model(self):
        pass


    def aggregate(self):
        super().aggregate()

        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            self.train_generator()
            self.train_global_inference_model()
            self.test_global_inference_model()


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

        diversity_loss_fn = Gen_DiversityLoss(metric='l1').to(self.device)
        div_beta = 0.0

        self.logger.log("[Server] Training Image-Based Global Generator...")
        for epoch in tqdm(range(self.global_gen_epochs), colour="blue", ncols=100):
            epoch_loss = 0
            
            batch_ids = random.choices(all_global_ids, k=self.batch_size)
            labels_input = torch.tensor(batch_ids).to(self.device)
            z = torch.randn(self.batch_size, self.gen_noise_dim).to(self.device)
            gen_imgs = self.global_gen(z, labels_input)

            flat_gen_imgs = gen_imgs.view(gen_imgs.size(0), -1)
            div_loss = diversity_loss_fn(flat_gen_imgs, z)
            
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

                target_list = []
                for gid in batch_ids:
                    if gid in global_to_local:
                        target_list.append(global_to_local[gid])
                    else:
                        target_list.append(-1)

                target_tensor = torch.tensor(target_list).to(self.device)
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

            if count_expert > 0:
                batch_loss_expert /= count_expert
            if count_observer > 0:
                batch_loss_observer /= count_observer
            
            total_loss = batch_loss_expert + (self.gen_observer_weight * batch_loss_observer) + (div_beta * div_loss)

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
        all_global_ids = list(all_global_ids)
        num_global_classes = len(all_global_ids)
        
        if num_global_classes == 0:
            self.logger.log("Warning: No mappings found via local_id_to_global_id. Skipping global model training.")
            return

        self.logger.log(f"Detected {num_global_classes} global classes: {sorted(list(all_global_ids))}")

        if self.model is None:
            self.logger.log(f"Initializing Global Inference Model with {num_global_classes} classes...")
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_global_classes, global_dim=256)
            self.global_model_optimizer = eval(self.global_model_optim_name)(self.model.parameters(), self.global_model_optim_lr)

        self.model.to(self.device)
        self.model.train()
        self.global_gen.eval() 

        criterion = nn.CrossEntropyLoss()

        y_list = []
        for c in range(num_global_classes):
            y_list.extend([c] * self.global_samples_per_class)

        for epoch in tqdm(range(self.global_model_epochs), colour="green", ncols=100):
            epoch_loss = 0
            random.shuffle(y_list)

            for i in range(0, len(y_list), self.batch_size):
                y_batch = torch.tensor(y_list[i:i+self.batch_size], dtype=torch.long).to(self.device)
                curr_batch_size = y_batch.size(0)
                z = torch.randn(curr_batch_size, self.gen_noise_dim).to(self.device)
                with torch.no_grad():
                    gen_imgs = self.global_gen(z, y_batch)
                
                self.global_model_optimizer.zero_grad()
                
                _, logits = self.model(gen_imgs)

                loss = criterion(logits, y_batch)
                loss.backward()
                self.global_model_optimizer.step()
                
                epoch_loss += loss.item()
            
            self.logger.log(f"Epoch {epoch} Loss: {epoch_loss:.4f}", print_to_console=False)
    

    def get_dir_soft_label(self, num_classes, batch_size):
        z = torch.randn(batch_size, self.gen_noise_dim, device=self.device)
        z = z.abs()
        z = torch.clamp(z, min=2.0)

        dir_alpha_niose = z.mean(dim=1, keepdim=True)
        dirichlet_alpha = dir_alpha_niose.expand(batch_size, num_classes)

        dist = torch.distributions.Dirichlet(dirichlet_alpha)
        dir_soft_labels = dist.sample()
        #print(dir_soft_labels)
        return dir_soft_labels


    def save_model(self, fname='checkpoints.pth'):
        self.logger.log("Saving UDON Server Checkpoints ...")

        client_label_distributions = {}
        for client in self.clients:
            unique_labels = set()
            for _, labels in client.train_loader:
                unique_labels.update(labels.tolist())
            client_label_distributions[client.id] = list(unique_labels)

        checkpoint = {
            'generator': self.global_gen.state_dict() if hasattr(self, 'global_gen') else None,
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