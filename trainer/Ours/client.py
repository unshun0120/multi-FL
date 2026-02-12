import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..BaseFL.client import Client as BaseClient
from utils.nets import ConditionalGenerator


class Client(BaseClient):
    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)

        self.feat_gen_noise_dim = kwargs.get('feat_gen_noise_dim', 128) 

        self.global_feature_gen = ConditionalGenerator(
            num_global_classes=len(self.local_label_to_global_id),
            noise_dim=self.feat_gen_noise_dim,
            output_dim=self.global_feature_dim 
        ).to(self.device)
        
        self.num_local_feat_gen_noise = kwargs.get('num_local_feat_gen_noise', 10)

        self.local_relation_weight = kwargs.get('local_relation_weight', 1.0)

    def update(self):
        self.model.to(self.device)
        self.model.train()

        if self.global_feature_gen is not None:
            self.global_feature_gen.eval()

        for epoch in range(self.local_epochs):
            epoch_loss = 0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                B = imgs.size(0)
                
                self.optimizer.zero_grad()

                local_feat, logits = self.model(imgs)

                # --- Classification Loss ---
                loss_ce = self.loss_fn(logits, labels)

                # --- Relation Distillation ---
                loss_distill = 0
                if self.global_feature_gen is not None:
                    with torch.no_grad():
                        best_gen_feats = []
                        
                        for i in range(B):
                            local_feat_i = local_feat[i].unsqueeze(0)
                            label_i = labels[i].unsqueeze(0)
                            z_i = torch.randn(self.num_local_feat_gen_noise, self.feat_gen_noise_dim).to(self.device)

                            label_i_extend = label_i.repeat(self.num_local_feat_gen_noise)
                            g_feat_candidates = self.global_feature_gen(z_i, label_i_extend) 

                            local_feat_i_norm = F.normalize(local_feat_i, dim=1)
                            g_feat_cands_norm = F.normalize(g_feat_candidates, dim=1)

                            similarity = torch.matmul(local_feat_i_norm, g_feat_cands_norm.T)
                            best_idx = torch.argmax(similarity)
                            best_feat = g_feat_candidates[best_idx]
                            best_gen_feats.append(best_feat)

                        teacher_features = torch.stack(best_gen_feats)
                        
                    student_norm = F.normalize(local_feat, dim=1)
                    teacher_norm = F.normalize(teacher_features, dim=1)   
                    relation_student = torch.matmul(student_norm, student_norm.T)
                    relation_teacher = torch.matmul(teacher_norm, teacher_norm.T)
                    
                    loss_distill = F.mse_loss(relation_student, relation_teacher)

                if self.global_feature_gen is None:
                    total_loss = loss_ce
                else: 
                    total_loss = loss_ce + self.local_relation_weight * loss_distill

                total_loss.backward()
                self.optimizer.step()
                epoch_loss += total_loss.item()
            
            self.round_train_loss = epoch_loss / len(self.train_loader)
        



