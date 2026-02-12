import torch
from torch import nn
import torch.nn.functional as F


class VanillaKDLoss(nn.Module):
    """ According to: Distilling the Knowledge in a Neural Network,
        https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, temperature):
        super(VanillaKDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        loss = F.kl_div(F.log_softmax(student_logits / self.temperature, dim=-1),
                        F.softmax(teacher_logits / self.temperature, dim=-1),
                        reduction='batchmean') * self.temperature * self.temperature
        return loss
    
class Gen_DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
    

"""
UDON Loss Functions
- Classification loss with margin (ArcFace, CosFace, NormFace)
- Logits distillation loss (KL divergence)
- Embedding similarity distillation loss (MSE on batch similarities)
"""

class ArcFaceMargin(nn.Module):
    """ArcFace margin transformation for cosine logits"""
    def __init__(self, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
    
    def forward(self, logits, targets):
        """
        Args:
            logits: cosine similarity logits (B, C)
            targets: ground truth labels (B,)
        """
        one_hot = F.one_hot(targets, logits.size(-1)).float()
        
        # ArcFace: add margin to angle
        theta = torch.acos(torch.clamp(logits * one_hot.sum(dim=1, keepdim=True), -1+1e-7, 1-1e-7))
        marginal_logits = torch.cos(theta + self.margin) * one_hot + logits * (1 - one_hot)
        
        return marginal_logits * self.scale


class CosFaceMargin(nn.Module):
    """CosFace margin transformation"""
    def __init__(self, scale=64.0, margin=0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
    
    def forward(self, logits, targets):
        one_hot = F.one_hot(targets, logits.size(-1)).float()
        marginal_logits = (logits - self.margin) * one_hot + logits * (1 - one_hot)
        return marginal_logits * self.scale


class NormFaceMargin(nn.Module):
    """NormFace (no margin, just scale)"""
    def __init__(self, scale=64.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, logits, targets):
        return logits * self.scale


class UDONLoss(nn.Module):
    def __init__(self, loss_config):
        super(UDONLoss, self).__init__()
        self.config = loss_config
        
        # Margin Loss Type
        self.loss_type = loss_config.get('type', 'normface')
        self.s = loss_config.get('scale', 30.0)
        self.m = loss_config.get('margin', 0.5)
        
        if self.loss_type == 'arcface':
            self.margin_fn = ArcFaceMargin(s=self.s, m=self.m)
        elif self.loss_type == 'cosface':
            self.margin_fn = CosFaceMargin(s=self.s, m=self.m)
        else:
            self.margin_fn = NormFaceMargin(s=self.s)
            
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, labels):
        """
        Calculates:
        1. Classification loss for Teacher
        2. Classification loss for Student
        3. Distillation loss (Logits KL)
        4. Distillation loss (Embedding Similarity)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Classification Losses
        # Transform logits using margin
        teacher_logits = self.margin_fn(outputs['teacher_logits'].clone(), labels)
        student_logits = self.margin_fn(outputs['universal_student_logits'].clone(), labels)
        
        loss_teacher = self.ce_loss(teacher_logits, labels)
        loss_student = self.ce_loss(student_logits, labels)
        
        w_t = self.config.get('classif_teacher_weight', 1.0)
        w_s = self.config.get('classif_student_weight', 1.0)
        
        total_loss += w_t * loss_teacher
        total_loss += w_s * loss_student
        
        loss_dict['loss_teacher'] = loss_teacher.item()
        loss_dict['loss_student'] = loss_student.item()
        
        # 2. Distillation: Logits (KL Divergence)
        if self.config.get('distill_logits', True):
            T = self.config.get('temperature', 1.0)
            
            # Detach teacher for distillation
            t_logits = outputs['teacher_logits'].detach() / T
            s_logits = outputs['universal_student_logits'] / T
            
            # KL(Student || Teacher)
            loss_distill_logits = F.kl_div(
                F.log_softmax(s_logits, dim=1),
                F.softmax(t_logits, dim=1),
                reduction='batchmean'
            ) * (T * T)
            
            w_d_l = self.config.get('distill_logits_weight', 1.0)
            total_loss += w_d_l * loss_distill_logits
            loss_dict['loss_distill_logits'] = loss_distill_logits.item()

        # 3. Distillation: Embeddings (Cosine Similarity or MSE)
        if self.config.get('distill_embeddings', True):
            t_emb = outputs['teacher_embedd'].detach()
            s_emb = outputs['universal_student_embedd']
            
            # Maximize cosine similarity -> Minimize 1 - cos_sim
            # Vectors are already L2 normalized in model
            cos_sim = (t_emb * s_emb).sum(dim=1).mean()
            loss_distill_embed = 1.0 - cos_sim
            
            w_d_e = self.config.get('distill_embed_weight', 1.0)
            total_loss += w_d_e * loss_distill_embed
            loss_dict['loss_distill_embed'] = loss_distill_embed.item()
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


"""
GeFL Loss Functions for generative models.
"""


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE ELBO loss = Reconstruction loss + β * KL divergence.
    
    Args:
        recon_x: Reconstructed samples
        x: Original samples
        mu: Mean of approximate posterior
        logvar: Log variance of approximate posterior
        beta: Weight for KL term (β-VAE)
    
    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss (per-sample)
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    
    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def gan_loss(d_real, d_fake, loss_type='bce'):
    """
    GAN loss for generator and discriminator.
    
    Args:
        d_real: Discriminator output for real samples
        d_fake: Discriminator output for fake samples
        loss_type: 'bce' for binary cross entropy, 'wgan' for Wasserstein
    
    Returns:
        (d_loss, g_loss)
    """
    if loss_type == 'bce':
        real_label = torch.ones_like(d_real)
        fake_label = torch.zeros_like(d_fake)
        
        # Discriminator loss: maximize log(D(x)) + log(1 - D(G(z)))
        d_loss_real = F.binary_cross_entropy(d_real, real_label)
        d_loss_fake = F.binary_cross_entropy(d_fake, fake_label)
        d_loss = d_loss_real + d_loss_fake
        
        # Generator loss: maximize log(D(G(z))) == minimize log(1 - D(G(z)))
        g_loss = F.binary_cross_entropy(d_fake, real_label)
        
    elif loss_type == 'wgan':
        # Wasserstein GAN loss
        d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
        g_loss = -torch.mean(d_fake)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return d_loss, g_loss


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for GAN training stability."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, real_features, fake_features):
        """
        Compute feature matching loss.
        
        Args:
            real_features: Features from discriminator for real samples
            fake_features: Features from discriminator for fake samples
        
        Returns:
            Feature matching loss
        """
        loss = 0
        for rf, ff in zip(real_features, fake_features):
            loss += F.mse_loss(ff.mean(dim=0), rf.mean(dim=0).detach())
        return loss

