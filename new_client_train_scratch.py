import torch
import torch.nn as nn

from trainer.NewClient.BaseNewClient import BaseNewClientTrainer, get_base_parser
from utils.test_utils import evaluate_client
from utils.csv_logger import append_csv

class ScratchTrainer(BaseNewClientTrainer):
    def train_client(self, client, d_name, arch_name, num_classes, label_to_global_id):
        """Train client with purely local CrossEntropy loss (Baseline)."""
        client.model.to(self.args.device)
        client_lr = self.exp_conf.get('optim_kwargs', {}).get('lr', 1e-3)
        optimizer = torch.optim.Adam(client.model.parameters(), lr=client_lr)
        criterion_ce = nn.CrossEntropyLoss()

        acc = evaluate_client(client, self.args)
        acc_history = [acc]
        self.logger.log(f"    Epoch 0 (init) | Acc: {acc:.2f}%")
        append_csv(self.csv_path, 'baseline', d_name, arch_name, 0, combined_acc=acc)

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
            append_csv(self.csv_path, 'baseline', d_name, arch_name, epoch + 1, combined_acc=acc)

        return acc_history

if __name__ == "__main__":
    args = get_base_parser().parse_args()
    trainer = ScratchTrainer(mode_name="baseline", args=args)
    trainer.run()