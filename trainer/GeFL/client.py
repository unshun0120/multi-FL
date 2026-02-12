import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from trainer.BaseFL.client import Node
from utils.nets import CCVAE, vae_loss

class GeFLClient(Node):
    def __init__(self, node_id, args, dataset_name, train_loader, test_loader, model, class_names, model_name, logger, **kwargs):
        super(GeFLClient, self).__init__(node_id, args, dataset_name, train_loader, test_loader, model, class_names, model_name, logger, **kwargs)
        
        # --- GeFL Specific Settings ---
        self.latent_size = getattr(args, 'latent_size', 16)
        self.img_size = getattr(args, 'img_size', 32)
        self.num_classes = len(class_names)
        self.beta_vae = getattr(args, 'beta_vae', 1.0)
        
        # Initialize Generator
        self.generator = CCVAE(
            num_classes=self.num_classes,
            latent_size=self.latent_size,
            img_size=self.img_size,
            channels=getattr(args, 'channels', 3)
        ).to(self.device)
        
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=getattr(args, 'gen_lr', 1e-3))
        self.gen_local_epochs = getattr(args, 'gen_local_epochs', 5)
        self.aid_by_gen = getattr(args, 'aid_by_gen', True)
        self.gen_sample_ratio = getattr(args, 'gen_sample_ratio', 1.0) # Ratio of gen samples to train samples

    def set_generator(self, gen_state_dict):
        """Called by Server to broadcast Global Generator."""
        self.generator.load_state_dict(gen_state_dict)

    def train_generator(self):
        """Train CVAE on local real data."""
        self.generator.train()
        total_loss = 0
        
        for epoch in range(self.gen_local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_onehot = F.one_hot(y, self.num_classes).float()
                
                self.gen_optimizer.zero_grad()
                recon, mu, logvar = self.generator(x, y_onehot)
                loss, _, _ = vae_loss(recon, x, mu, logvar, self.beta_vae)
                
                loss.backward()
                self.gen_optimizer.step()
                total_loss += loss.item()
        
        return total_loss

    def train_target_model(self):
        """Train Heterogeneous Target Model using Real + Generated Data."""
        self.model.train()
        self.generator.eval()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        
        # Standard local epochs for target model
        epochs = self.args.local_epochs
        
        for epoch in range(epochs):
            batch_loss = []
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # --- 1. Real Data Step ---
                optimizer.zero_grad()
                pred_real = self.model(x)
                
                # Incase model returns tuple (feats, logits)
                if isinstance(pred_real, tuple): pred_real = pred_real[1]
                
                loss_real = criterion(pred_real, y)
                loss_real.backward()
                
                # --- 2. Generated Data Step (Data Augmentation) ---
                if self.aid_by_gen:
                    num_gen = int(x.size(0) * self.gen_sample_ratio)
                    with torch.no_grad():
                        x_gen, y_gen = self.generator.sample(num_gen, device=self.device)
                        # Ensure generated data is same scale as real (e.g., CVAE output tanh [-1,1], real might be normalized)
                        # Assuming simple case here.
                    
                    pred_gen = self.model(x_gen)
                    if isinstance(pred_gen, tuple): pred_gen = pred_gen[1]
                    
                    loss_gen = criterion(pred_gen, y_gen)
                    loss_gen.backward()
                
                optimizer.step()
                batch_loss.append(loss_real.item())
            
            losses.append(sum(batch_loss)/len(batch_loss))
            
        return sum(losses)/len(losses)

    def train(self):
        """
        GeFL Main Local Update:
        1. Train Generator (CVAE).
        2. Train Target Model (augmented by CVAE).
        3. Return Generator weights to Server (Target Model stays local).
        """
        # 1. Train Generator
        self.train_generator()
        
        # 2. Train Target Model
        loss = self.train_target_model()
        
        self.logger.info(f"Client {self.id} finished training. Loss: {loss:.4f}")
        
        # 3. Return Generator Weights (Target model is heterogeneous, not aggregated)
        # We assume server calls this to get weights
        return {
            'generator_weights': self.generator.state_dict(),
            'train_loss': loss,
            'num_samples': len(self.train_loader.dataset)
        }