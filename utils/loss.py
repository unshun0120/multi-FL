import torch
from torch import nn
import torch.nn.functional as F

class CAM_Hook:
    def __init__(self, module):
        # 攔截最後一個 Conv2d 層的輸出
        self.hook = module.register_forward_hook(self.hook_fn)
        self.activation_map = None

    def hook_fn(self, module, input, output):
        # output 維度: [Batch, Channels, Height, Width]
        # 我們取 Channels 維度的絕對值平均，得到二維的空間啟動強度圖 [Batch, 1, Height, Width]
        self.activation_map = output.abs().mean(dim=1, keepdim=True)

    def close(self):
        self.hook.remove()

def get_gaussian_mask(size, device, sigma=0.3):
    """
    """
    x = torch.linspace(-1, 1, size, device=device)
    y = torch.linspace(-1, 1, size, device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    mask = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return mask.unsqueeze(0).unsqueeze(0) # [1, 1, size, size]

def get_cam_loss(cam_hooks, target_mask):
    cam_loss = 0.0
    valid_hooks = 0
    for hook in cam_hooks:
        if hook.activation_map is not None:
            # 將 activation map 縮放到與 target_mask 一樣大 (例如 32x32 或更小)
            act_map_resized = F.interpolate(hook.activation_map, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
            
            # 正規化啟動圖到 0~1 之間
            act_min = act_map_resized.view(act_map_resized.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
            act_max = act_map_resized.view(act_map_resized.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
            act_norm = (act_map_resized - act_min) / (act_max - act_min + 1e-8)
            
            # MUSE 論文的 Margin Loss 精神：當啟動值沒有達到 Target 遮罩的強度時給予懲罰
            # L_cam = max(0, M_target - M(x))
            loss = F.relu(target_mask - act_norm).mean()
            cam_loss += loss
            valid_hooks += 1
            
    if valid_hooks > 0:
        return cam_loss / valid_hooks
    return 0.0

class BNSM_Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.running_mean = module.running_mean
        self.running_var = module.running_var
        self.batch_mean = None
        self.batch_var = None

    def hook_fn(self, module, input, output):
        self.batch_mean = input[0].mean([0, 2, 3])
        self.batch_var = input[0].var([0, 2, 3], unbiased=False)

    def close(self):
        self.hook.remove()

def get_bn_loss(bn_hooks):
    bn_loss = 0.0
    valid_hooks_count = 0
    for hook in bn_hooks:
        if hook.batch_mean is not None and hook.batch_var is not None:
            bn_loss += torch.norm(hook.batch_mean - hook.running_mean, 2)
            bn_loss += torch.norm(hook.batch_var - hook.running_var, 2)
            valid_hooks_count += 1
            
    if valid_hooks_count > 0:
        return bn_loss / valid_hooks_count
    return 0.0


def total_variation_loss(img):
    """計算相鄰像素的差異，讓生成的圖片變平滑，避免對抗性雜訊"""
    tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    # 除以總像素量做正規化
    return (tv_h + tv_w) / (img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3])


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
    

class RelationDistillationLoss(nn.Module):
    def __init__(self):
        super(RelationDistillationLoss, self).__init__()

    def forward(self, s_feat, t_feat):
        s_feat = F.normalize(s_feat, p=2, dim=1)
        t_feat = F.normalize(t_feat, p=2, dim=1)
        
        s_dist = torch.cdist(s_feat, s_feat, p=2)
        t_dist = torch.cdist(t_feat, t_feat, p=2)
        
        s_dist = s_dist / (s_dist.mean() + 1e-8)
        t_dist = t_dist / (t_dist.mean() + 1e-8)
        
        loss = F.mse_loss(s_dist, t_dist)
        return loss
    

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