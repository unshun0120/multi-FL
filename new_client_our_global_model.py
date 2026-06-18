import os
import torch
import torch.nn as nn
import argparse

from trainer.NewClient.BaseNewClient import BaseNewClientTrainer, get_base_parser, DATASET_META
from utils.csv_logger import append_csv
from utils.nets import get_heterogeneous_model

class FinetuneGlobalModelTrainer(BaseNewClientTrainer):
    def __init__(self, mode_name, args):
        super().__init__(mode_name, args)
        
        server_dir = os.path.dirname(self.args.model_path)
        self.global_model_path = os.path.join(server_dir, 'server_global_model.pth')
            
        self.global_model_state = torch.load(self.global_model_path, map_location=self.args.device)
        self.logger.log(f"[Init] Server Global Model parameters from: {self.global_model_path}")

    def train_client(self, client, d_name, arch_name, num_classes, label_to_global_id):
        server_arch_id = 3
        if d_name == "SuperDataset":
            in_ch, img_size = 3, 32
        else:
            in_ch = DATASET_META[d_name]['in_ch']
            img_size = DATASET_META[d_name]['size']
        
        client.model = get_heterogeneous_model(
            node_id=server_arch_id, 
            in_channels=in_ch, 
            num_classes=self.num_global_classes, 
            img_size=img_size, 
            global_dim=self.global_feature_dim
        )
        
        client.model.load_state_dict(self.global_model_state, strict=False)
        client.model.to(self.args.device)
        
        client_lr = self.exp_conf.get('optim_kwargs', {}).get('lr', 1e-3)
        optimizer = torch.optim.Adam(client.model.parameters(), lr=client_lr)
        criterion_ce = nn.CrossEntropyLoss()

        def evaluate_global_acc(model, test_loader):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(self.args.device)
                    global_labels = torch.tensor([label_to_global_id[lbl.item()] for lbl in labels], device=self.args.device)
                    
                    _, logits = model(imgs)
                    preds = logits.argmax(dim=1)
                    correct += (preds == global_labels).sum().item()
                    total += len(labels)
            return (correct / total * 100.0) if total > 0 else 0.0

        acc = evaluate_global_acc(client.model, client.test_loader)
        acc_history = [acc]
        self.logger.log(f"    Epoch 0 | Acc: {acc:.2f}%")
        #append_csv(self.csv_path, self.mode_name, d_name, arch_name, 0, combined_acc=acc)

        for epoch in range(self.args.new_client_epochs):
            client.model.train()
            epoch_ce, num_batches = 0.0, 0

            for imgs, labels in client.train_loader:
                imgs = imgs.to(self.args.device)
                global_labels = torch.tensor([label_to_global_id[lbl.item()] for lbl in labels], device=self.args.device)
                
                optimizer.zero_grad()
                _, logits = client.model(imgs)
                
                loss_ce = criterion_ce(logits, global_labels)
                loss_ce.backward()
                optimizer.step()
                
                epoch_ce += loss_ce.item()
                num_batches += 1

            acc = evaluate_global_acc(client.model, client.test_loader)
            acc_history.append(acc)
            
            self.logger.log(f"    Epoch {epoch+1}/{self.args.new_client_epochs} | CE Loss: {epoch_ce/max(num_batches, 1):.4f} | Acc: {acc:.2f}%")
            #append_csv(self.csv_path, self.mode_name, d_name, arch_name, epoch + 1, combined_acc=acc)

        return acc_history

if __name__ == "__main__":
    args = get_base_parser().parse_args()
    trainer = FinetuneGlobalModelTrainer(mode_name="our_finetune_global", args=args)
    trainer.run()