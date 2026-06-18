import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import random
import clip
import kornia.augmentation as Kaug

# -------------------------
# DFKD General Utils
# -------------------------

def evaluate_student_model(student_model, test_loader, device):
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, logits = student_model(inputs) 
            
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    return (correct / total) * 100.0 if total > 0 else 0.0

def KLDiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

def JSDiv(logits_s, logits_t, T=3.0):
    P = F.softmax(logits_s / T, dim=1)
    Q = F.softmax(logits_t / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction='batchmean') + 0.5 * F.kl_div(torch.log(Q), M, reduction='batchmean')

# -------------------------
# DeepInversion Utils
# -------------------------

def jitter_and_flip(inputs_jit, lim=1./8., do_flip=True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)

    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + \
                  (diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1

class DeepInversionHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.r_feature = 0.0

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        self.r_feature = torch.norm(module.running_var.data - var, 2) + \
                         torch.norm(module.running_mean.data - mean, 2)

    def remove(self):
        self.hook.remove()


# -------------------------
# FAST Utils
# -------------------------

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).to(p.device)
        p.grad.data.add_(p.data - tar_p.data, alpha=67)

def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).to(p.device)
        p.grad.data.add_(tar_p.grad.data)

def get_fast_augmentation(img_size=(32, 32)):
    return nn.Sequential(
        Kaug.RandomCrop(size=img_size, padding=4, padding_mode="reflect"),
        Kaug.RandomHorizontalFlip()
    )

# -------------------------
# NAYER Utils
# -------------------------

def nayer_cross_entropy(logits, targets, cr=0.5):
    ys = torch.zeros_like(logits)
    ys.fill_(cr / (ys.shape[1] - 1))
    ys.scatter_(1, targets.unsqueeze(1), (1 - cr))
    log_preds = F.log_softmax(logits, dim=1)
    cls_loss = -(ys * log_preds).sum(dim=1).mean()
    return cls_loss

def get_nayer_label_embedding(class_names, device):
    model, _ = clip.load("ViT-B/32", device=device)
    model = model.to(device)
    model.eval()

    text_prompts = []
    for l in class_names:
        l_str = str(l).replace("_", " ").replace("-", " ")
        prompt = f"a image of {l_str}"
        text_prompts.append(prompt)
            
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features.float() 
        
    return text_features