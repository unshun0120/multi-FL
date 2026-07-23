"""
Client for GeFL (DeepInversion / Data-Free KD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from tqdm import tqdm

from trainer.BaseFL.client import Client as BaseClient

class Client(BaseClient):
    def __init__(self, **exp_conf):
        super(Client, self).__init__(**exp_conf)

        self.img_size = exp_conf.get('img_size', 32)
        self.channels = exp_conf.get('channels', 3)


    def update(self):
        """
        Overrides BaseFL Node update() method.
        Local update:
        1) Train local target model
        2) Upload local model weights to Server for DeepInversion
        """
        self.round_train_loss = self.train_target_model()

        self.logger.log(
            f"Client {self.id} ({self.dataset_name}) | Target Loss: {self.round_train_loss:.4f}"
        )

        return {
            "local_model": self.model.state_dict(),
            "num_samples": self.num_samples,
        }
    

    def train_target_model(self):
        """Train local heterogeneous target model using real data."""
        self.model.train()
        self.model.to(self.device)

        epoch_losses = []

        for _ in range(self.local_epochs):
            batch_losses = []
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.local_optimizer.zero_grad()

                pred_real = self.model(x)
                if isinstance(pred_real, tuple):
                    pred_real = pred_real[1]
                loss = self.local_loss_fn(pred_real, y)


                loss.backward()
                self.local_optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))

        return sum(epoch_losses) / max(1, len(epoch_losses))
    