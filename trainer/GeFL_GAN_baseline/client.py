"""
Client for GeFL GAN baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from tqdm import tqdm

from trainer.BaseFL.client import Client as BaseClient
from utils.nets import DCGANGenerator, DCGANDiscriminator

class Client(BaseClient):
    def __init__(self, **exp_conf):
        super(Client, self).__init__(**exp_conf)

        self.img_size = exp_conf.get('img_size', 32)
        self.channels = exp_conf.get('channels', 3)
        self.noise_dim = exp_conf.get('gen_noise_dim', 128)

        self.gen_local_epochs = exp_conf.get('gen_local_epochs', 5)
        self.gen_sample_ratio = exp_conf.get('gen_sample_ratio', 1.0)
        self.gan_lr = exp_conf.get('gen_lr', 2e-4)
        self.gan_beta1 = exp_conf.get('gan_beta1', 0.5)
        self.gan_beta2 = exp_conf.get('gan_beta2', 0.999)

        self.generator = DCGANGenerator(
            num_classes=self.local_num_classes,
            noise_dim=self.noise_dim,
            img_size=self.img_size,
            channels=self.channels,
        ).to(self.device)

        self.discriminator = DCGANDiscriminator(
            num_classes=self.local_num_classes,
            img_size=self.img_size,
            channels=self.channels,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.gan_lr, betas=(self.gan_beta1, self.gan_beta2))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.gan_lr, betas=(self.gan_beta1, self.gan_beta2))

        self.adv_criterion = nn.BCEWithLogitsLoss()


    def update(self):
        """
        Overrides BaseFL Node update() method.
        GeFL local update:
        1) Train local DCGAN (G,D)
        2) Train target model aided by G
        """
        gan_stat = self.train_generator()
        self.round_train_loss = self.train_target_model()

        self.logger.log(
            f"Client {self.id} ({self.dataset_name}) | Target Loss: {self.round_train_loss:.4f} | "
            f"G Loss: {gan_stat['g_loss']:.4f} | D Loss: {gan_stat['d_loss']:.4f}",
        )

        return {
            "generator_weights": self.generator.state_dict(),
            "discriminator_weights": self.discriminator.state_dict(),
            "num_samples": self.num_samples,
        }
    

    def train_target_model(self):
        """Train local heterogeneous target model using real + generated data."""
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
    

    def train_generator(self):
        """Local DCGAN training (train G and D)."""
        self.generator.train()
        self.discriminator.train()

        g_loss_total, d_loss_total, n_steps = 0.0, 0.0, 0

        for _ in range(self.gen_local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                bsz = x.size(0)

                real_t = torch.ones(bsz, 1, device=self.device)
                fake_t = torch.zeros(bsz, 1, device=self.device)

                # 1) Train D
                self.d_optimizer.zero_grad()

                real_logit = self.discriminator(x, y).view(-1, 1)
                d_real = self.adv_criterion(real_logit, real_t)

                z = torch.randn(bsz, self.noise_dim, device=self.device)
                x_fake = self.generator(z, y).detach()
                fake_logit = self.discriminator(x_fake, y).view(-1, 1)
                d_fake = self.adv_criterion(fake_logit, fake_t)

                d_loss = 0.5 * (d_real + d_fake)
                d_loss.backward()
                self.d_optimizer.step()

                # 2) Train G
                self.g_optimizer.zero_grad()
                z = torch.randn(bsz, self.noise_dim, device=self.device)
                x_fake = self.generator(z, y)
                fake_logit = self.discriminator(x_fake, y).view(-1, 1)

                g_loss = self.adv_criterion(fake_logit, real_t)
                g_loss.backward()
                self.g_optimizer.step()

                d_loss_total += d_loss.item()
                g_loss_total += g_loss.item()
                n_steps += 1

        return {
            "g_loss": g_loss_total / max(1, n_steps),
            "d_loss": d_loss_total / max(1, n_steps),
        }
    

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