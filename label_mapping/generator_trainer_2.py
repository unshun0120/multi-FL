import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import math
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

from utils.nets import ConditionalImageGenerator, NLGenerator
from utils.DFKD_utils import (
    # deepinversion
    KLDiv,
    JSDiv,
    jitter_and_flip,
    get_image_prior_losses,
    DeepInversionHook,
    # fast
    reptile_grad,
    fomaml_grad,
    # nayer
    get_nayer_label_embedding,
)

GEN_CONFIG = {
    'img_num_samples': 64,
    'feat_gen_noise_dim': 128,
    'gen_epochs': 1000,
    'gen_lr': 5e-4,
    'student_lr': 1e-3,
    'kd_steps': 1,     
}


# -------------------------
# DeepInversion
# -------------------------

di_weight = {
    'epochs': 20, 
    'g_steps': 500, 
    'lr': 0.01,
    'ce': 1.0,      
    'bn': 0.05,      
    'tv': 0.05,    
    'l2': 0.001,    
    'adv': 1.0, 
}

def train_generators_DeepInversion(clients_dict, label_space_meta, dataset_meta, device, logger, 
                                   use_new_gen_method=True, client_label_mask_dict=None, student_model=None):
    
    method_name = "New Gen Method" if use_new_gen_method else "Old Gen Method"
    logger.log(f"[Testing] Training Per-Dataset Generators Offline ({method_name})...")

    generators_dict = {}

    for ls_id, dataset_clients in clients_dict.items():
        if len(dataset_clients) == 0:
            continue

        current_student = student_model[ls_id] if (student_model is not None and ls_id in student_model) else None

        if current_student is not None:
            student_optimizer = torch.optim.Adam(current_student.parameters(), lr=GEN_CONFIG['student_lr'])
    
        class_names = label_space_meta[ls_id]
        num_local_classes = len(class_names)

        dataset_clients.sort(key=lambda m: m.client_id)
        
        if use_new_gen_method and client_label_mask_dict is not None and ls_id in client_label_mask_dict:
            raw_mask = client_label_mask_dict[ls_id]
            current_mask = raw_mask[:len(dataset_clients)].to(device)
            valid_client_indices = [i for i in range(len(dataset_clients)) if current_mask[i].sum() > 0]
        else:
            current_mask = torch.ones((len(dataset_clients), num_local_classes), device=device)
            valid_client_indices = list(range(len(dataset_clients)))

        batch_size = 64
        epoch_loss_tracker = []
        
        all_client_bn_hooks = []
        for client_model in dataset_clients:
            client_model.eval() 
            for param in client_model.parameters():      
                param.requires_grad = False

            hooks = []
            for module in client_model.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False
                if isinstance(module, torch.nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(module))
            all_client_bn_hooks.append(hooks)

        valid_client_count = len(valid_client_indices)

        if valid_client_count > 0:
            all_valid_labels = torch.where(current_mask[valid_client_indices].sum(dim=0) > 0)[0]
        else:
            all_valid_labels = torch.arange(num_local_classes, device=device)

        class_image_pools = {c: [] for c in range(num_local_classes)}
        image_pool = deque(maxlen=2000)

        for epoch in tqdm(range(di_weight['epochs']), colour='blue', ncols=100, desc=f"Gen:{ls_id}"):
            
            if current_student is not None:
                current_student.eval()

            batch_labels = all_valid_labels[torch.randint(0, len(all_valid_labels), (batch_size,)).to(device)]

            inputs = torch.randn(batch_size, 3, 32, 32, device=device).requires_grad_()
            pixel_optimizer = torch.optim.Adam([inputs], lr=di_weight['lr'], betas=[0.5, 0.99])

            best_cost = 1e6
            best_inputs = None

            for it in range(di_weight['g_steps']):
                pixel_optimizer.zero_grad() 
                inputs_aug = jitter_and_flip(inputs, lim=1./8., do_flip=False)
                
                batch_loss = 0.0
                clients_contributed = 0 

                for client_idx in valid_client_indices:
                    client_model = dataset_clients[client_idx]
                    client_mask = current_mask[client_idx] 
                    
                    known_mask = client_mask[batch_labels].bool() 
                    
                    if not known_mask.any():
                        continue 
                    
                    sub_imgs = inputs_aug[known_mask]
                    sub_labels = batch_labels[known_mask]
                    
                    _, logits_t = client_model(sub_imgs)
                    cls_loss = F.cross_entropy(logits_t, sub_labels)
                    
                    bn_loss = 0.0
                    client_hooks = all_client_bn_hooks[client_idx]
                    if len(client_hooks) > 0 and di_weight['bn'] != 0:
                        bn_loss = sum([h.r_feature for h in client_hooks]) / len(client_hooks)
                    
                    sub_imgs_prior = inputs[known_mask]
                    tv_loss = get_image_prior_losses(sub_imgs_prior)
                    l2_loss = torch.norm(sub_imgs_prior, 2)

                    client_loss = (di_weight['ce'] * cls_loss) + \
                                    (di_weight['bn'] * bn_loss) + \
                                    (di_weight['tv'] * tv_loss) + \
                                    (di_weight['l2'] * l2_loss)
                    
                    if current_student is not None and di_weight['adv'] > 0:
                        _, logits_s = current_student(sub_imgs)
                        loss_adv = -JSDiv(logits_s, logits_t.detach(), T=3.0)
                        client_loss += di_weight['adv'] * loss_adv
                    
                    batch_loss = batch_loss + client_loss
                    clients_contributed += 1

                if clients_contributed > 0:
                    avg_loss = batch_loss / clients_contributed
                    avg_loss.backward()
                    pixel_optimizer.step()

                    inputs.data = inputs.data.clamp_(-1.0, 1.0)
                    
                    if avg_loss.item() < best_cost:
                        best_cost = avg_loss.item()
                        best_inputs = inputs.detach()

            if best_inputs is not None:
                epoch_loss_tracker.append(best_cost)

                for i in range(batch_size):
                    lbl = batch_labels[i].item()
                    class_image_pools[lbl].append(best_inputs[i].detach().cpu())

                for client_idx in valid_client_indices:
                    client_mask = current_mask[client_idx] 
                    known_mask = client_mask[batch_labels].bool() 
                    if known_mask.any():
                        sub_imgs = best_inputs[known_mask]
                        image_pool.append((client_idx, sub_imgs.detach()))

            if current_student is not None and len(image_pool) > 0:
                current_student.train()
                
                for _ in range(GEN_CONFIG.get('kd_steps', 400)):
                    student_optimizer.zero_grad()
                    for _ in range(valid_client_count):
                        pool_idx = random.randint(0, len(image_pool) - 1)
                        p_client_idx, p_imgs = image_pool[pool_idx]
                        p_client_model = dataset_clients[p_client_idx]
                        
                        with torch.no_grad():
                            _, logits_t = p_client_model(p_imgs)
                        
                        _, logits_s = current_student(p_imgs)
                        kd_loss = KLDiv(logits_s, logits_t).mean()
                        (kd_loss / valid_client_count).backward()
                    student_optimizer.step()

        if len(epoch_loss_tracker) > 0:
            avg_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
            logger.log(f"    Dataset [{ls_id}] Generator | Epochs: {di_weight['epochs']} | Final Loss: {avg_loss:.4f}")

        for hooks in all_client_bn_hooks:
            for h in hooks:
                if hasattr(h, 'close'): h.close()
                elif hasattr(h, 'remove'): h.remove()

        for c in range(num_local_classes):
            if len(class_image_pools[c]) > 0:
                class_image_pools[c] = torch.stack(class_image_pools[c])
            else:
                class_image_pools[c] = torch.randn(GEN_CONFIG['img_num_samples'], 3, 32, 32)

        generators_dict[ls_id] = class_image_pools

        # Draw
        draw_name = '_gen_img_new' if use_new_gen_method else '_gen_img_old'
        draw_synthetic_samples('DeepInversion', draw_name, class_image_pools, class_names, logger, ls_id, device)

    return generators_dict


# -------------------------
# FAST
# -------------------------

fast_weight = {
    'epochs': 120,
    'g_steps': 5,
    'lr': GEN_CONFIG.get('gen_lr', 5e-4),
    'ce': 1.0,      
    'bn': 1.0,      
    'adv': 1.0, 
}

def train_generators_FAST(clients_dict, label_space_meta, dataset_meta, device, logger, 
                          use_new_gen_method=True, client_label_mask_dict=None, student_model=None):
    
    method_name = "New Gen Method" if use_new_gen_method else "Old Gen Method"
    logger.log(f"[Testing] Training Per-Dataset Generators Offline ({method_name}) using FAST...")

    generators_dict = {}
    is_maml = True

    for ls_id, dataset_clients in clients_dict.items():
        if len(dataset_clients) == 0:
            continue
            
        current_student = student_model[ls_id] if (student_model is not None and ls_id in student_model) else None

        if current_student is not None:
             student_optimizer = torch.optim.Adam(current_student.parameters(), lr=GEN_CONFIG['student_lr'])

        class_names = label_space_meta[ls_id]
        num_local_classes = len(class_names)
        dataset_clients.sort(key=lambda m: m.client_id)
        
        if use_new_gen_method and client_label_mask_dict is not None and ls_id in client_label_mask_dict:
            raw_mask = client_label_mask_dict[ls_id]
            current_mask = raw_mask[:len(dataset_clients)].to(device)
            valid_client_indices = [i for i in range(len(dataset_clients)) if current_mask[i].sum() > 0]
        else:
            current_mask = torch.ones((len(dataset_clients), num_local_classes), device=device)
            valid_client_indices = list(range(len(dataset_clients)))

        generator = ConditionalImageGenerator(
            num_classes=num_local_classes, 
            noise_dim=GEN_CONFIG['feat_gen_noise_dim'],
            img_channels=3, 
            img_size=32   
        ).to(device)

        meta_optimizer = torch.optim.Adam(generator.parameters(), lr=fast_weight['lr'] * fast_weight['g_steps'], betas=[0.5, 0.999])
        generator.train()

        batch_size = 64
        epoch_loss_tracker = []
        
        all_client_bn_hooks = []
        for client_model in dataset_clients:
            client_model.eval() 
            for param in client_model.parameters():      
                param.requires_grad = False

            hooks = []
            for module in client_model.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False
                if isinstance(module, torch.nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(module))
            all_client_bn_hooks.append(hooks)

        valid_client_count = len(valid_client_indices)
        image_pool = deque(maxlen=2000)

        if valid_client_count > 0:
            all_valid_labels = torch.where(current_mask[valid_client_indices].sum(dim=0) > 0)[0]
        else:
            all_valid_labels = torch.arange(num_local_classes, device=device)

        for epoch in tqdm(range(fast_weight['epochs']), colour='green', ncols=100, desc=f"FAST-Gen:{ls_id}"):
            meta_optimizer.zero_grad() 
            total_epoch_loss = 0.0
            epoch_best_images = []

            if current_student is not None:
                current_student.eval()

            z_noise = torch.randn(batch_size, GEN_CONFIG['feat_gen_noise_dim'], device=device).requires_grad_()
            batch_labels = all_valid_labels[torch.randint(0, len(all_valid_labels), (batch_size,)).to(device)]
            
            fast_generator = ConditionalImageGenerator(num_local_classes, GEN_CONFIG['feat_gen_noise_dim'], 3, 32).to(device)
            fast_generator.load_state_dict(generator.state_dict())

            inner_optimizer = torch.optim.Adam([
                {'params': fast_generator.parameters()},
                {'params': [z_noise], 'lr': 0.01}
            ], lr=fast_weight['lr'], betas=[0.5, 0.999])

            best_cost = 1e6
            best_inputs_global = None

            for it in range(fast_weight['g_steps']):
                gen_imgs = fast_generator(z_noise, batch_labels)
                gen_imgs_aug = jitter_and_flip(gen_imgs, lim=1./8., do_flip=True)
                
                batch_loss = 0.0
                clients_contributed = 0

                for client_idx in valid_client_indices:
                    client_model = dataset_clients[client_idx]
                    client_mask = current_mask[client_idx]
                    
                    known_mask = client_mask[batch_labels].bool()
                    if not known_mask.any():
                        continue

                    sub_imgs = gen_imgs_aug[known_mask]
                    sub_labels = batch_labels[known_mask]
                    
                    _, logits_t = client_model(sub_imgs)
                    cls_loss = F.cross_entropy(logits_t, sub_labels)
                    
                    bn_loss = 0.0
                    client_hooks = all_client_bn_hooks[client_idx]
                    if len(client_hooks) > 0 and fast_weight['bn'] != 0:
                        bn_loss = sum([h.r_feature for h in client_hooks]) / len(client_hooks)

                    client_loss = (fast_weight['ce'] * cls_loss) + (fast_weight['bn'] * bn_loss) 
                    
                    if current_student is not None:
                        _, logits_s = current_student(sub_imgs)
                        mask = (logits_s.max(1)[1] == logits_t.max(1)[1]).float()
                        loss_adv = -(KLDiv(logits_s, logits_t.detach(), reduction='none').sum(1) * mask).mean()
                        client_loss += fast_weight['adv'] * loss_adv
                    
                    batch_loss = batch_loss + client_loss
                    clients_contributed += 1

                if clients_contributed > 0:
                    avg_batch_loss = batch_loss / clients_contributed
                    
                    inner_optimizer.zero_grad()
                    avg_batch_loss.backward()

                    if avg_batch_loss.item() < best_cost:
                        best_cost = avg_batch_loss.item()
                        best_inputs_global = gen_imgs.detach()

                    if is_maml:
                        if it == 0: meta_optimizer.zero_grad()
                        fomaml_grad(generator, fast_generator)
                        if it == (fast_weight['g_steps'] - 1): meta_optimizer.step()

                    inner_optimizer.step()
            
            if best_inputs_global is not None:
                epoch_loss_tracker.append(best_cost)
                for client_idx in valid_client_indices:
                    client_mask = current_mask[client_idx]
                    known_mask = client_mask[batch_labels].bool()
                    if known_mask.any():
                        sub_imgs = best_inputs_global[known_mask]
                        epoch_best_images.append((client_idx, sub_imgs.detach()))

            if not is_maml:
                meta_optimizer.zero_grad()
                reptile_grad(generator, fast_generator)
                meta_optimizer.step()

            for c_idx, b_imgs in epoch_best_images:
                image_pool.append((c_idx, b_imgs))

            if current_student is not None and len(image_pool) > 0:
                current_student.train()
                generator.eval() 
                
                for _ in range(GEN_CONFIG.get('kd_steps', 400)):
                    student_optimizer.zero_grad()
                    for _ in range(valid_client_count):
                        pool_idx = random.randint(0, len(image_pool) - 1)
                        p_client_idx, p_imgs = image_pool[pool_idx]
                        p_client_model = dataset_clients[p_client_idx]
                        
                        with torch.no_grad():
                            _, logits_t = p_client_model(p_imgs)
                        
                        _, logits_s = current_student(p_imgs)
                        kd_loss = KLDiv(logits_s, logits_t).mean()
                        
                        (kd_loss / valid_client_count).backward()
                    student_optimizer.step()
                    
                generator.train()
                    
        if len(epoch_loss_tracker) > 0:
            avg_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
            logger.log(f"    Dataset [{ls_id}] Generator | Epochs: {fast_weight['epochs']} | Final Loss: {avg_loss:.4f}")

        for hooks in all_client_bn_hooks:
            for h in hooks:
                if hasattr(h, 'close'): h.close()
                elif hasattr(h, 'remove'): h.remove()

        generators_dict[ls_id] = generator

        # Draw
        draw_name = '_gen_img_new' if use_new_gen_method else '_gen_img_old'
        draw_synthetic_samples('FAST', draw_name, generator, class_names, logger, ls_id, device)

    return generators_dict 


# -------------------------
# NAYER
# -------------------------

nayer_weight = {
    'epochs': 120,       
    'g_loops': 2,         
    'g_steps': 40,      
    'g_life': 10,        
    'kd_steps': 400,     
    'lr_g': 4e-3,        
    'ce': 0.5,      
    'bn': 10.0,      
    'adv': 1.33,
}

def train_generators_NAYER(clients_dict, label_space_meta, dataset_meta, device, logger, 
                           use_new_gen_method=True, client_label_mask_dict=None, student_model=None):
    
    method_name = "New Gen Method" if use_new_gen_method else "Old Gen Method"
    logger.log(f"[Testing] Training Per-Dataset Generators Offline ({method_name}) using NAYER...")

    generators_dict = {}

    for ls_id, dataset_clients in clients_dict.items():
        if len(dataset_clients) == 0:
            continue
            
        current_student = student_model[ls_id] if (student_model is not None and ls_id in student_model) else None

        if current_student is not None:
             student_optimizer = torch.optim.Adam(current_student.parameters(), lr=GEN_CONFIG['student_lr'])

        class_names = label_space_meta[ls_id]
        num_local_classes = len(class_names)
        dataset_clients.sort(key=lambda m: m.client_id)
        
        if use_new_gen_method and client_label_mask_dict is not None and ls_id in client_label_mask_dict:
            raw_mask = client_label_mask_dict[ls_id]
            current_mask = raw_mask[:len(dataset_clients)].to(device)
            valid_client_indices = [i for i in range(len(dataset_clients)) if current_mask[i].sum() > 0]
        else:
            current_mask = torch.ones((len(dataset_clients), num_local_classes), device=device)
            valid_client_indices = list(range(len(dataset_clients)))

        logger.log(f"    -> Loading CLIP ViT-B/32 label embeddings for {num_local_classes} classes...")
        real_label_emb = get_nayer_label_embedding(class_names, device)

        batch_size = 64
        epoch_loss_tracker = []

        generator = NLGenerator(
            ngf=64,
            img_size=32, 
            nc=3, 
            nl=100,            
            label_emb=real_label_emb, 
            le_emb_size=256, 
            le_size=512,        
            sbz=batch_size
        ).to(device)

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=nayer_weight['lr'], betas=[0.5, 0.999])
        generator.train()

        all_client_bn_hooks = []
        for client_model in dataset_clients:
            client_model.eval()
            for param in client_model.parameters():      
                param.requires_grad = False
            hooks = []
            for module in client_model.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False
                if isinstance(module, torch.nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(module))
            all_client_bn_hooks.append(hooks)

        valid_client_count = len(valid_client_indices)
        image_pool = deque(maxlen=2000)

        if valid_client_count > 0:
            all_valid_labels = torch.where(current_mask[valid_client_indices].sum(dim=0) > 0)[0]
        else:
            all_valid_labels = torch.arange(num_local_classes, device=device)

        for epoch in tqdm(range(nayer_weight['epochs']), colour='magenta', ncols=100, desc=f"NAYER-Gen:{ls_id}"):
            
            if epoch > 0 and epoch % nayer_weight['g_life'] == 0:
                generator = generator.reinit()
            
            if current_student is not None:
                current_student.eval()
            
            for gloop in range(nayer_weight['g_loops']):
                generator.re_init_le()

                gen_optimizer = torch.optim.Adam(generator.parameters(), lr=nayer_weight['lr_g'], betas=[0.5, 0.999])

                batch_labels = all_valid_labels[torch.randint(0, len(all_valid_labels), (batch_size,)).to(device)]
                best_cost = 1e6
                best_inputs = None
                
                for it in range(nayer_weight['g_steps']):
                    gen_optimizer.zero_grad()
                    gen_imgs = generator(targets=batch_labels)
                    gen_imgs_aug = jitter_and_flip(gen_imgs, lim=1./8., do_flip=True)
                    batch_loss = 0.0
                    clients_contributed = 0

                    for client_idx in valid_client_indices:
                        client_model = dataset_clients[client_idx]
                        client_mask = current_mask[client_idx]
                        
                        known_mask = client_mask[batch_labels].bool()
                        if not known_mask.any():
                            continue
                        
                        sub_imgs = gen_imgs_aug[known_mask]
                        sub_labels = batch_labels[known_mask]

                        _, logits_t = client_model(sub_imgs)
                        cls_loss = F.cross_entropy(logits_t, sub_labels)
                        
                        bn_loss = 0.0
                        client_hooks = all_client_bn_hooks[client_idx]
                        if len(client_hooks) > 0 and nayer_weight['bn'] != 0:
                            bn_loss = sum([h.r_feature for h in client_hooks]) / len(client_hooks)

                        client_loss = (nayer_weight['ce'] * cls_loss) + (nayer_weight['bn'] * bn_loss)
                        
                        if current_student is not None and nayer_weight['adv'] > 0:
                            _, logits_s = current_student(sub_imgs)
                            mask = (logits_s.max(1)[1] == logits_t.max(1)[1]).float()
                            kl_val = F.kl_div(F.log_softmax(logits_s, dim=1), F.softmax(logits_t.detach(), dim=1), reduction='none').sum(1)
                            loss_adv = -(kl_val * mask).mean()
                            client_loss += nayer_weight['adv'] * loss_adv
                        
                        batch_loss = batch_loss + client_loss
                        clients_contributed += 1

                    if clients_contributed > 0:
                        avg_loss = batch_loss / clients_contributed
                        avg_loss.backward()
                        gen_optimizer.step()

                        if avg_loss.item() < best_cost:
                            best_cost = avg_loss.item()
                            best_inputs = gen_imgs.detach()

            if best_inputs is not None:
                for client_idx in valid_client_indices:
                    client_mask = current_mask[client_idx]
                    known_mask = client_mask[batch_labels].bool()
                    if known_mask.any():
                        sub_imgs = best_inputs[known_mask]
                        image_pool.append((client_idx, sub_imgs.detach()))
            
        if current_student is not None and len(image_pool) > 0:
            current_student.train()
            generator.eval() 
            
            for _ in range(GEN_CONFIG.get('kd_steps', 400)):
                student_optimizer.zero_grad()
                for _ in range(valid_client_count):
                    pool_idx = random.randint(0, len(image_pool) - 1)
                    p_client_idx, p_imgs = image_pool[pool_idx]
                    p_client_model = dataset_clients[p_client_idx]
                    
                    with torch.no_grad():
                        _, logits_t = p_client_model(p_imgs)
                    
                    _, logits_s = current_student(p_imgs)
                    kd_loss = KLDiv(logits_s, logits_t).mean()
                    
                    (kd_loss / valid_client_count).backward()
                student_optimizer.step()
            generator.train()

        if len(epoch_loss_tracker) > 0:
            avg_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
            logger.log(f"    Dataset [{ls_id}] Generator | Epochs: {nayer_weight['epochs']} | Final Loss: {avg_loss:.4f}")

        for hooks in all_client_bn_hooks:
            for h in hooks:
                if hasattr(h, 'close'): h.close()
                elif hasattr(h, 'remove'): h.remove()

        generators_dict[ls_id] = generator

        # Draw
        draw_name = '_gen_img_new' if use_new_gen_method else '_gen_img_old'
        draw_nayer_samples('NAYER', draw_name, generator, class_names, logger, ls_id, device)

    return generators_dict


def draw_synthetic_samples(method, fname, generator_or_pool, class_names, logger, ls_id, device):
    with torch.no_grad():
        save_dir = logger.log_dir
        num_samples_per_class = 1 
        num_local_classes = len(class_names)
        num_classes_to_plot = num_local_classes

        if isinstance(generator_or_pool, dict):
            # Deepinversion
            sample_imgs = []
            for c in range(num_classes_to_plot):
                if c in generator_or_pool and len(generator_or_pool[c]) > 0:
                    sample_imgs.append(generator_or_pool[c][0].to(device))
                else:
                    sample_imgs.append(torch.randn(3, 32, 32).to(device))
            sample_imgs = torch.stack(sample_imgs)
        else:
            # FAST
            generator_or_pool.eval()
            sample_labels = torch.arange(num_classes_to_plot).repeat(num_samples_per_class).to(device)
            sample_z = torch.randn(num_samples_per_class * num_classes_to_plot, GEN_CONFIG['feat_gen_noise_dim']).to(device)
            sample_imgs = generator_or_pool(sample_z, sample_labels)

        sample_imgs = (sample_imgs * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

        cols = min(10, num_classes_to_plot)
        rows = math.ceil(num_classes_to_plot / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes_flat = [axes] if rows * cols == 1 else axes.flatten()
        
        for idx in range(num_classes_to_plot):
            ax = axes_flat[idx]
            img = sample_imgs[idx] 
            img = np.transpose(img, (1, 2, 0))
            ax.imshow(img)
            ax.set_title(str(class_names[idx]), fontsize=12)
            ax.axis('off') 

        for idx in range(num_classes_to_plot, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{method}_{ls_id}_{fname}.png')
        plt.savefig(save_path)
        plt.close() 
        logger.log(f"-> [New] Gen images saved to {save_path}")


def draw_nayer_samples(method, fname, generator, class_names, logger, ls_id, device):
    
    generator.eval()

    with torch.no_grad():
        save_dir = logger.log_dir
        num_local_classes = len(class_names)
        num_classes_to_plot = num_local_classes
        sample_labels = torch.arange(num_classes_to_plot).to(device)
        
        generator.re_init_le()
        sample_imgs = generator(targets=sample_labels)
        sample_imgs = (sample_imgs * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

        cols = min(10, num_classes_to_plot)
        rows = math.ceil(num_classes_to_plot / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes_flat = [axes] if rows * cols == 1 else axes.flatten()
        
        for idx in range(num_classes_to_plot):
            ax = axes_flat[idx]
            ax.imshow(sample_imgs[idx].transpose(1, 2, 0))
            ax.set_title(class_names[idx], fontsize=8)
            ax.axis('off')

        for idx in range(num_classes_to_plot, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{method}_{ls_id}_{fname}.png')
        plt.savefig(save_path)
        plt.close() 
        logger.log(f"-> [New] Gen images saved to {save_path}")