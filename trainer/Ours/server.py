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

from trainer.BaseFL.server import Server as Base_Server
from utils.nets import ConditionalGenerator, Classifier

class Server(Base_Server):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.feat_gen_noise_dim = kwargs.get('feat_gen_noise_dim', 128)

        self.global_feature_gen = ConditionalGenerator(
            num_global_classes=len(self.local_label_to_global_id),
            noise_dim=self.feat_gen_noise_dim,
            output_dim=self.global_feature_dim 
        ).to(self.device)

        self.global_feat_gen_epochs = kwargs.get('global_feat_gen_epochs', 10) 
        self.feat_gen_optim_lr = kwargs.get('feat_gen_optim_lr', 1e-3)
        self.feat_gen_optim_name =  kwargs.get('feat_gen_optim', 'Adam')
        self.feat_gen_optimizer = eval(self.feat_gen_optim_name)(self.global_feature_gen.parameters(), self.feat_gen_optim_lr)

        self.shared_classifier = Classifier(
            input_dim=self.global_feature_dim,
            num_classes=self.num_global_classes
        ).to(self.device)
        
        self.shared_cls_epochs = kwargs.get('shared_cls_epochs', 10) 
        self.shared_cls_optim_lr = kwargs.get('shared_cls_optim_lr', 1e-3)
        self.shared_cls_optim_name =  kwargs.get('shared_cls_optim', 'Adam')
        self.shared_cls_optimizer = eval(self.shared_cls_optim_name)(self.shared_classifier.parameters(), self.shared_cls_optim_lr)
         
        self.feat_gen_observer_weight = kwargs.get('feat_gen_observer_weight', 1.0) 

    def aggregate(self):
        groups = defaultdict(list)
        for client in self.selected_clients:
            ls_id = str(client.class_name_set)
            groups[ls_id].append(client)
            
            if ls_id not in self.label_space_meta:
                self.label_space_meta[ls_id] = client.class_name_set

            # print(ls_id)
            # e.g. ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            # ["Ankle boot", "Bag", "Coat", "Dress", "Pullover", "Sandal", "Shirt", "Sneaker", "T-shirt/top", "Trouser"]

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} label spaces)...") 
        
        for ls_id, group_clients in groups.items():
            if ls_id not in self.global_models:
                self.global_models[ls_id] = {}

            # aggregate clients generic classifier
            msg_list = [(client.num_samples, client.model.classifier.state_dict())
                        for client in group_clients]
            w_cls = self.avg_weights(msg_list)

            num_classes = w_cls['weight'].shape[0]
            input_dim = w_cls['weight'].shape[1]

            cls_model = Classifier(input_dim, num_classes).to(self.device)
            cls_model.load_state_dict(w_cls)
            cls_model.eval()

            self.global_models[ls_id]['classifier'] = cls_model

        self.train_generator()

        self.train_global_shared_classifier_hard_label()

    def train_generator(self):
        self.global_feature_gen.train()

        all_classifiers = {}
        for ls_id, model_dict in self.global_models.items():
            if 'classifier' in model_dict:
                all_classifiers[ls_id] = model_dict['classifier']

        all_global_ids = list(self.global_id_to_local_label.keys())

        self.logger.log("[Server] Training Feature-Based Global Generator...")
        for epoch in tqdm(range(self.global_feat_gen_epochs), colour="blue"):
            epoch_loss = 0
            random.shuffle(all_global_ids)
            
            for i in range(0, len(all_global_ids), self.batch_size):
                batch_ids = all_global_ids[i : i + self.batch_size]
                curr_batch = len(batch_ids)

                labels_input = torch.tensor(batch_ids).to(self.device)
                z = torch.randn(curr_batch, self.feat_gen_noise_dim).to(self.device)
                gen_feat = self.global_feature_gen(z, labels_input)
                
                count_expert = 0
                count_observer = 0

                batch_loss_expert = 0
                batch_loss_observer = 0
                
                self.feat_gen_optimizer.zero_grad()

                for ls_id, classifier in all_classifiers.items():
                    logits = classifier(gen_feat)

                    group_class_names = self.label_space_meta[ls_id]

                    # 看這個classifier認不認識batch裡的global_id
                    target_list = []
                    for gid in batch_ids:
                        g_name = self.global_id_to_local_label[gid]
                        if g_name in group_class_names:
                            # 認識 -> 存他資料集原本的local id
                            target_list.append(group_class_names.index(g_name))
                        else:
                            # 不認識 -> 存-1
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
                
                total_loss = batch_loss_expert + (self.feat_gen_observer_weight * batch_loss_observer)

                total_loss.backward()
                self.feat_gen_optimizer.step()
                
                epoch_loss += total_loss.item()

            self.logger.log(f"  Epoch {epoch+1}/{self.global_feat_gen_epochs} | Loss: {epoch_loss:.4f}", print_to_console=False)

        self.logger.log("[Server] Feature-Based Global Generator training finished.")

    def train_global_shared_classifier_hard_label(self):
        self.global_feature_gen.eval()       
        self.shared_classifier.train() 

        all_global_ids = list(self.global_id_to_local_label.keys())

        self.logger.log(f"[Server] Training Shared Classifier - Hard Label...")
        for epoch in tqdm(range(self.shared_cls_epochs)):
            epoch_loss = 0
            random.shuffle(all_global_ids)

            for i in range(0, len(all_global_ids), self.batch_size):
                batch_ids = all_global_ids[i : i + self.batch_size]
                current_bs = len(batch_ids)
                
                labels_input = torch.tensor(batch_ids).to(self.device)
                z = torch.randn(current_bs, self.feat_gen_noise_dim).to(self.device)
                
                with torch.no_grad():
                    gen_feat = self.global_feature_gen(z, labels_input)
                
                self.shared_cls_optimizer.zero_grad()
                
                logits = self.shared_classifier(gen_feat)
                loss = F.cross_entropy(logits, labels_input)
                
                loss.backward()
                self.shared_cls_optimizer.step()
                
                epoch_loss += loss.item()
            
            self.logger.log(f"  [Hard Label Shared Classifier] Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    def distribute_model(self):
        gen_state_dict = None
        if self.global_feature_gen is not None:
            gen_state_dict = self.global_feature_gen.state_dict()

        for client in self.selected_clients:
            ls_id = str(client.class_name_set)
            
            if ls_id not in self.global_models:
                continue

            global_part = self.global_models[ls_id]

            if 'classifier' in global_part:
                client.model.classifier.load_state_dict(global_part['classifier'])

            if gen_state_dict is not None:
                if hasattr(client, 'global_feature_gen') and client.global_feature_gen is not None:
                    client.global_feature_gen.load_state_dict(gen_state_dict)
    

    def get_dir_soft_label(self, num_classes, batch_size):
        z = torch.randn(batch_size, self.feat_gen_noise_dim, device=self.device)
        z = z.abs()
        z = torch.clamp(z, min=1.0)

        dir_alpha_niose = z.mean(dim=1, keepdim=True)
        dirichlet_alpha = dir_alpha_niose.expand(batch_size, num_classes)

        dist = torch.distributions.Dirichlet(dirichlet_alpha)
        dir_soft_labels = dist.sample()

        return dir_soft_labels

