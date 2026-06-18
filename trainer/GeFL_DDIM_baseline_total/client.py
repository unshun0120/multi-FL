import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from tqdm import tqdm

from trainer.BaseFL.client import Client as BaseClient
from utils.nets import ContextUnet, DDIM

class Client(BaseClient):
    def __init__(self, **exp_conf):
        super(Client, self).__init__(**exp_conf)

        self.img_size = exp_conf.get('img_size', 32)
        self.channels = exp_conf.get('channels', 3)

        self.gen_local_epochs = exp_conf.get('gen_local_epochs', 5)
        self.aid_by_gen = exp_conf.get('aid_by_gen', False)
        self.gen_sample_ratio = exp_conf.get('gen_sample_ratio', 1.0)
        self.gen_lr = exp_conf.get('gen_lr', 2e-4)

        n_feat = exp_conf.get('n_feat', 64) 
        unet = ContextUnet(in_channels=self.channels, n_feat=n_feat, n_classes=self.local_num_classes)
        
        self.ddpm = DDIM(
            nn_model=unet, 
            betas=(1e-4, 0.02), 
            n_T=1000,
            device=self.device,
            drop_prob=0.1
        ).to(self.device)

        self.gen_optimizer = torch.optim.Adam(self.ddpm.parameters(), lr=self.gen_lr)


    def update(self):
        """
        Overrides BaseFL Node update() method.
        GeFL local update:
        1) Train local DCGAN (G,D)
        2) Train target model aided by G
        """
        gen_loss = self.train_generator()
        self.round_train_loss = self.train_target_model()

        self.logger.log(
            f"Client {self.id} ({self.dataset_name}) | Target Loss: {self.round_train_loss:.4f} | "
            f"DDPM Loss: {gen_loss:.4f}"
        )

        return {
            "generator_weights": self.ddpm.state_dict(),
            "num_samples": self.num_samples,
        }
    

    def train_target_model(self):
        """Train local heterogeneous target model using real + generated data."""
        self.model.train()
        self.model.to(self.device)
        self.ddpm.eval()

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

                if self.aid_by_gen:
                    n_gen = max(1, int(x.size(0) * self.gen_sample_ratio))
                    x_gen, y_gen = self.sample_generated(n_gen)

                    pred_gen = self.model(x_gen)
                    if isinstance(pred_gen, tuple):
                        pred_gen = pred_gen[1]
                    loss = loss + self.local_loss_fn(pred_gen, y_gen)

                loss.backward()
                self.local_optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))

        return sum(epoch_losses) / max(1, len(epoch_losses))
    

    def train_generator(self):
        self.ddpm.train()
        g_loss_total, n_steps = 0.0, 0

        for _ in range(self.gen_local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.gen_optimizer.zero_grad()
                
                loss = self.ddpm(x, y)
                
                loss.backward()
                self.gen_optimizer.step()

                g_loss_total += loss.item()
                n_steps += 1

        return g_loss_total / max(1, n_steps)
    

    def sample_generated(self, n):
        y_gen = torch.randint(0, self.local_num_classes, (n,), device=self.device)
        with torch.no_grad():
            x_gen, _ = self.ddpm.sample(n_sample=n, size=(self.channels, self.img_size, self.img_size), device=self.device, guide_w=3.0,)
        return x_gen, y_gen
    

    def get_avg_features(self):
        self.model.eval()
        self.model.to(self.device)
        features_dict = {}
        
        with torch.no_grad():
            for imgs, labels in self.train_loader: 
                imgs = imgs.to(self.device)
                feats, _ = self.model(imgs)
                
                for i in range(len(labels)):
                    lbl = labels[i].item()
                    if lbl not in features_dict:
                        features_dict[lbl] = []
                    features_dict[lbl].append(feats[i])
        
        avg_features = {}
        for lbl, feats in features_dict.items():
            avg_features[lbl] = torch.stack(feats).mean(dim=0).unsqueeze(0)
            
        return avg_features