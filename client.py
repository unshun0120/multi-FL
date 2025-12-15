import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class Client:
    def __init__(self, client_id, args, train_dataset, test_dataset, model, class_names, model_name):
        self.client_id = client_id
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model.to(args.device)
        self.class_names = class_names 
        self.model_name = model_name

        if args.optim == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.gen_lr, weight_decay=1e-4)
        else:    
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.gen_lr, momentum=0.9)

        self.criterion = nn.CrossEntropyLoss()
        self.global_generator = None

    def local_train(self):
        self.model.train()

        if self.global_generator is not None:
            self.global_generator.eval()

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        for epoch in range(self.args.local_epochs):
            epoch_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)
                B = imgs.size(0)
                
                self.optimizer.zero_grad()

                # local_global_feat: adapter layer feature(dim=256)
                local_global_feat, logits = self.model(imgs)

                # --- Classification Loss ---
                loss_ce = self.criterion(logits, labels)

                # --- Relation Distillation ---
                loss_distill = 0
                if self.global_generator is not None:
                    with torch.no_grad():
                        # 存batch中每個local feature最相似的generator feature
                        best_gen_feats = []
                        
                        for i in range(B):
                            local_feat_i = local_global_feat[i].unsqueeze(0)
                            label_i = labels[i].unsqueeze(0)
                            z_i = torch.randn(self.args.num_local_noise, self.args.noise_dim).to(self.args.device)

                            label_i_extend = label_i.repeat(self.args.num_local_noise)
                            g_feat_candidates = self.global_generator(z_i, label_i_extend) 

                            local_feat_i_norm = F.normalize(local_feat_i, dim=1)
                            g_feat_cands_norm = F.normalize(g_feat_candidates, dim=1)

                            # 算similarity
                            similarity = torch.matmul(local_feat_i_norm, g_feat_cands_norm.T)
                            # 找出similarity值最大(最相似)的index
                            best_idx = torch.argmax(similarity)
                            best_feat = g_feat_candidates[best_idx]
                            best_gen_feats.append(best_feat)

                        teacher_features = torch.stack(best_gen_feats)
                        
                    student_norm = F.normalize(local_global_feat, dim=1)
                    teacher_norm = F.normalize(teacher_features, dim=1)   
                    relation_student = torch.matmul(student_norm, student_norm.T)
                    relation_teacher = torch.matmul(teacher_norm, teacher_norm.T)
                    
                    loss_distill = F.mse_loss(relation_student, relation_teacher)

                if self.global_generator is None:
                    total_loss = loss_ce
                else: 
                    total_loss = loss_ce + self.args.local_relation_weight * loss_distill

                total_loss.backward()
                self.optimizer.step()
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss

    def update_local_model(self, global_classifier_state_dict):
        # 拿global classifier直接蓋掉local classifier 
        if global_classifier_state_dict is not None:
            self.model.classifier.load_state_dict(global_classifier_state_dict)

    def test(self):
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)
                
                _, logits = self.model(imgs)
                
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = 100.0 * correct / total if total > 0 else 0.0
        return acc
    