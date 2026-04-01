import torch
import torch.nn as nn
import os
import argparse
from importlib import import_module

from trainer.NewClient.BaseNewClient import BaseNewClientTrainer, get_base_parser
from utils.test_utils import evaluate_client
from utils.csv_logger import append_csv
from utils.nets import get_heterogeneous_model

class FinetuneGlobalModelTrainer(BaseNewClientTrainer):
    def __init__(self, mode_name, args):
        super().__init__(mode_name, args)
        
        global_model_path = os.path.join(self.save_dir, 'global_inference_model.pth')
        self.global_model_state = torch.load(global_model_path, map_location=self.args.device)
        self.logger.log(f"[Init] Loaded Global Model state from {global_model_path}")

        # 這裡假設 Global Model 的架構我們需要知道。
        # 通常 Server 會有一個固定的架構 (e.g., node_id=3 -> ResNet18)
        # 這裡我們暫時 hardcode 或者是從 exp_conf 讀取 Server 架構 ID
        # 假設 Server 用的是 ResNet18 (id=3) 作為 Global Model，你可以根據實際情況修改
        # 或者我們直接在 train_client 覆蓋 client.model
        pass

    def train_client(self, client, d_name, arch_name, num_classes, label_to_global_id):
        """
        Train client by FINE-TUNING the Global Model.
        """
        # =========================================================================
        # 步驟 1: 將 Client Model 替換為 Global Model
        # =========================================================================
        # 我們必須確保 Client Model 的架構跟 Global Model 一致才能載入權重
        # 這裡我們做一個比較粗略的嘗試：直接把 global_model_state 載入 client.model
        # 如果架構不對，Load State Dict 會報錯。
        
        
        # 嘗試載入 Global Weights
        # strict=False 允許一些不匹配 (例如 classifier head 維度不同)
        # 但因為我們是 global_inference_model，它的 head 應該已經是 global_num_classes 了
        # 而 client.model 初始化時可能是 local_num_classes 或者是 global_num_classes
        # 在 BaseNewClient.py 裡，get_heterogeneous_model 傳入的是 d_meta['classes'] (Local)
        # 所以這裡 Head 維度會對不上！
        
        # 修正 Client Model 的 Head 以匹配 Global Model (Global Classes)
        if client.model.classifier.out_features != self.num_global_classes:
            in_features = client.model.classifier.in_features
            client.model.classifier = nn.Linear(in_features, self.num_global_classes)
            client.model.to(self.args.device)
        
        client.model.load_state_dict(self.global_model_state, strict=True)
        self.logger.log(f"    -> Successfully loaded Global Model weights into {arch_name}")

        client.model.to(self.args.device)
        
        client_lr = self.exp_conf.get('optim_kwargs', {}).get('lr', 1e-4)
        optimizer = torch.optim.Adam(client.model.parameters(), lr=client_lr)
        criterion_ce = nn.CrossEntropyLoss()

        acc = evaluate_client(client, self.args)
        acc_history = [acc]
        self.logger.log(f"    Epoch 0 (Global Model Zero-shot) | Acc: {acc:.2f}%")
        append_csv(self.csv_path, 'our_finetune_global', d_name, arch_name, 0, combined_acc=acc)

        for epoch in range(self.args.new_client_epochs):
            client.model.train()
            epoch_ce, num_batches = 0.0, 0

            for imgs, labels in client.train_loader:
                imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)
                
                optimizer.zero_grad()
                _, logits = client.model(imgs)
                
                loss_ce = criterion_ce(logits, labels)
                
                loss_ce.backward()
                optimizer.step()
                epoch_ce += loss_ce.item()
                num_batches += 1

            acc = evaluate_client(client, self.args)
            acc_history.append(acc)
            self.logger.log(f"    Epoch {epoch+1}/{self.args.new_client_epochs} | CE: {epoch_ce/max(num_batches, 1):.4f} | Acc: {acc:.2f}%")
            append_csv(self.csv_path, 'our_finetune_global', d_name, arch_name, epoch + 1, combined_acc=acc)

        return acc_history

if __name__ == "__main__":
    args = get_base_parser().parse_args()
    trainer = FinetuneGlobalModelTrainer(mode_name="our_finetune_global", args=args)
    trainer.run()