import torch
import copy
from collections import defaultdict
import json
from torch.nn import *
from torch.optim import *
import torch.nn.functional as F
from tqdm import tqdm
import random
import os
import numpy as np

from trainer.BaseFL.server import Server as BaseServer
from utils.nets import Classifier
from utils.loss import RelationDistillationLoss
from utils.train_utils import evaluate_model
from data.datasets import get_readable_class_names

DATA_ROOT = './data/raw'

class Server(BaseServer):
    def __init__(self, public_dataloader=None, **exp_conf):
        super(Server, self).__init__(**exp_conf)
        self.public_dataloader = public_dataloader

        self.shared_classifier = Classifier(
            input_dim=self.global_feature_dim,
            num_classes=self.num_global_classes
        ).to(self.device)
        
        self.num_public_train_teacher = exp_conf.get('num_public_train_teacher', None)

        self.shared_layers = ['backbone', 'universal_projection']
        
        self.rd_loss_fn = RelationDistillationLoss()

    def aggregate(self):
        """
        Apply UDON into FL, this work doesn't need to aggregate
        """
        self.collect_client_models = defaultdict(list)

        for client in self.selected_clients:
            ls_id = json.dumps(sorted(list(client.class_name_set)))
            
            fe_copy = copy.deepcopy(client.model.feature_extractor).cpu()
            if hasattr(client.model, 'adapter'):
                adapter_copy = copy.deepcopy(client.model.adapter).cpu()
            else:
                adapter_copy = None
            cls_copy = copy.deepcopy(client.model.classifier).cpu()
            
            client_data = {
                'client_id': client.id, 
                'dataset_name': client.dataset_name, 
                'feature_extractor': fe_copy,
                'adapter': adapter_copy,
                'classifier': cls_copy,
                'num_samples': client.num_samples
            }
            
            self.collect_client_models[ls_id].append(client_data)
        
        self.train_on_public_dataset()

    def train_on_public_dataset(self):
        print("Server: Training on Public Dataset...")
        self.model.to(self.device)
        self.model.train()

        dataset_indices_cache = {}
        
        for epoch in range(self.global_full_model_epochs):
            self.logger.log(f"--- Epoch {epoch+1}/{self.global_full_model_epochs} ---")

            for ls_id, client_list in self.collect_client_models.items():
                class_names = json.loads(ls_id)
                dataset_name = client_list[0].get('dataset_name')
                
                current_loader = self.public_train_loaders[dataset_name]

                if dataset_name not in dataset_indices_cache:
                    local_class_names = get_readable_class_names(dataset_name, DATA_ROOT)
                    
                    global_indices = [self.local_label_to_global_id[name] for name in local_class_names]
                    
                    dataset_indices_cache[dataset_name] = torch.tensor(global_indices).to(self.device)

                global_indices_tensor = dataset_indices_cache[dataset_name]
                
                if self.num_public_train_teacher is not None and len(client_list) > self.num_public_train_teacher:
                    selected_clients_data = random.sample(client_list, self.num_public_train_teacher)
                    self.logger.log(f">>> Distilling {dataset_name}: Sampling {self.num_public_train_teacher}/{len(client_list)} teachers...")
                else:
                    selected_clients_data = client_list
                    self.logger.log(f">>> Distilling {dataset_name}: Using all {len(client_list)} teachers...")

                active_teachers = []
                for c_data in selected_clients_data: 
                    teacher = {
                        'fe': c_data['feature_extractor'].to(self.device).eval(),
                        'cls': c_data['classifier'].to(self.device).eval(),
                        'adapter': c_data['adapter'].to(self.device).eval() if c_data['adapter'] else None
                    }
                    active_teachers.append(teacher)

                total_loss = 0.0
                batch_count = 0

                for batch_idx, (images, _) in enumerate(current_loader):
                    images = images.to(self.device)
                    
                    s_feat, s_logits = self.model(images)
                    
                    s_logits_subset = s_logits[:, global_indices_tensor]

                    t_logits_sum = 0.0
                    t_feat_sum = 0.0
                    valid_teachers = 0

                    for teacher in active_teachers:
                        with torch.no_grad():
                            feat = teacher['fe'](images)
                            if len(feat.shape) > 2: 
                                feat = torch.flatten(feat, 1)

                            if teacher['adapter'] is not None:
                                feat = teacher['adapter'](feat)
                            
                            logits = teacher['cls'](feat)

                            if logits.shape[1] > len(global_indices_tensor):
                                # 使用相同的索引進行切片，確保對齊
                                logits = logits[:, global_indices_tensor]
                        
                        t_feat_sum += feat
                        t_logits_sum += logits
                        valid_teachers += 1

                    if valid_teachers == 0: continue

                    t_feat_avg = t_feat_sum / valid_teachers
                    t_logits_avg = t_logits_sum / valid_teachers

                    loss_kd = self.kd_loss_fn(s_logits_subset, t_logits_avg)
                    loss_rd = self.rd_loss_fn(s_feat, t_feat_avg)
                    
                    loss = 1.0 * loss_kd + 0.1 * loss_rd
                    
                    self.global_full_model_optimizer.zero_grad()
                    loss.backward()
                    self.global_full_model_optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    print(f"   [{dataset_name}] Avg Loss: {total_loss / batch_count:.4f}")

                for teacher in active_teachers:
                    teacher['fe'].to('cpu')
                    teacher['cls'].to('cpu')
                    if teacher['adapter']: teacher['adapter'].to('cpu')
                
                torch.cuda.empty_cache()

        self.model.to('cpu')
        print("[Server] Dataset-Specific Distillation Finished.")


    def distribute_model(self):
        """
        Distribute only shared layers to clients.
        Clients keep their private layers (Teacher, Classifiers).
        """
        pass

    def evaluate_generic(self):
        self.model.eval()
        self.model.to(self.device)
        
        acc_list, loss_list = [], []
        criterion = torch.nn.CrossEntropyLoss()

        self.logger.log("\n[Server] Evaluating on Public Test Sets...")

        for d_name, test_loader in self.public_test_loaders.items():
            local_class_names = get_readable_class_names(d_name, DATA_ROOT)
            global_indices = [self.local_label_to_global_id[name] for name in local_class_names]
            global_indices_tensor = torch.tensor(global_indices).to(self.device)
            
            total_loss = 0.0
            correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    _, logits = self.model(x)
                    
                    logits_subset = logits[:, global_indices_tensor]
                    
                    loss = criterion(logits_subset, y)
                    total_loss += loss.item() * y.size(0)
                    
                    predicted = logits_subset.argmax(dim=1)
                    correct += predicted.eq(y).sum().item()
                    total_samples += y.size(0)

            if total_samples > 0:
                d_acc = correct / total_samples
                d_loss = total_loss / total_samples
                
                acc_list.append(d_acc)
                loss_list.append(d_loss)
                
                self.logger.log(f"   >>> {d_name:<12}: Acc = {d_acc*100:.2f}% | Loss = {d_loss:.4f}")

        if len(acc_list) > 0:
            self.g_acc = np.mean(acc_list)
            self.g_loss = np.mean(loss_list)
        else:
            self.g_acc = 0.0
            self.g_loss = 0.0
            
        self.logger.log(f"[Server] Global Avg Acc: {self.g_acc*100:.2f}%\n")

        # 釋放 GPU 資源
        self.model.to('cpu')
        torch.cuda.empty_cache()

    def save_model(self, fname='feat-generator.pth'):
        torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, fname))