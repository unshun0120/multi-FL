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
    'gen_epochs': 1500, 
    'lr': GEN_CONFIG.get('gen_lr', 5e-4),
    'ce': 0.5,      
    'bn': 10.0,      
    'tv': 1e-5,    
    'l2': 0.0,    
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

        generator = ConditionalImageGenerator(
            num_classes=num_local_classes, 
            noise_dim=GEN_CONFIG['feat_gen_noise_dim'],
            img_channels=3, 
            img_size=32   
        ).to(device)

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=di_weight['lr'])
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

        #current_student = None

        for epoch in tqdm(range(di_weight['gen_epochs']), colour='blue', ncols=100, desc=f"Gen:{ls_id}"):
            gen_optimizer.zero_grad() 
            total_epoch_loss = 0.0
            total_adv_loss = 0.0

            if current_student is not None:
                current_student.eval()

            epoch_images = []

            for client_idx in valid_client_indices:
                client_model = dataset_clients[client_idx]
                client_mask = current_mask[client_idx]
                
                valid_labels = torch.where(client_mask == 1)[0]
                if len(valid_labels) == 0: continue
                
                batch_labels = valid_labels[torch.randint(0, len(valid_labels), (batch_size,)).to(device)]
                z_noise = torch.randn(batch_size, GEN_CONFIG['feat_gen_noise_dim']).to(device)
                
                gen_imgs = generator(z_noise, batch_labels)
                epoch_images.append((client_idx, gen_imgs.detach()))

                #gen_imgs = jitter_and_flip(gen_imgs, lim=1./8., do_flip=True)
                
                _, logits_t = client_model(gen_imgs)
                cls_loss = F.cross_entropy(logits_t, batch_labels)
                
                bn_loss = 0.0
                client_hooks = all_client_bn_hooks[client_idx]
                if len(client_hooks) > 0 and di_weight['bn'] != 0:
                    bn_loss = sum([h.r_feature for h in client_hooks]) / len(client_hooks)
                
                tv_loss = get_image_prior_losses(gen_imgs)
                l2_loss = torch.norm(gen_imgs, 2)

                client_loss = (di_weight['ce'] * cls_loss) + \
                                (di_weight['bn'] * bn_loss) + \
                                (di_weight['tv'] * tv_loss) + \
                                (di_weight['l2'] * l2_loss)
                
                if current_student is not None:
                    _, logits_s = current_student(gen_imgs)
                    loss_adv = -JSDiv(logits_s, logits_t.detach(), T=3.0)

                    client_loss += di_weight['adv'] * loss_adv
                    total_adv_loss += loss_adv.item()
                
                (client_loss / valid_client_count).backward()
                total_epoch_loss += client_loss.item()

            for c_idx, imgs in epoch_images:
                image_pool.append((c_idx, imgs))
            
            if valid_client_count > 0:
                gen_optimizer.step()
                epoch_loss_tracker.append(total_epoch_loss / valid_client_count)

            if current_student is not None and len(image_pool) > 0:
                current_student.train()
                generator.eval() 
                
                for _ in range(GEN_CONFIG['kd_steps']):
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
            logger.log(f"     Dataset [{ls_id}] Generator | Epochs: {di_weight['gen_epochs']} | Final Loss: {avg_loss:.4f}")

        for hooks in all_client_bn_hooks:
            for h in hooks:
                if hasattr(h, 'close'): h.close()
                elif hasattr(h, 'remove'): h.remove()

        generators_dict[ls_id] = generator

        # Draw
        draw_name = '_gen_img_new' if use_new_gen_method else '_gen_img_old'
        draw_synthetic_samples('DeepInversion', draw_name, generator, class_names, logger, ls_id, device)

    return generators_dict


# -------------------------
# FAST
# -------------------------

fast_weight = {
    'gen_epochs': 1500, 
    'lr': GEN_CONFIG.get('gen_lr', 5e-4),
    'ce': 0.5,      
    'bn': 10.0,      
    'adv': 1.0, 
}

def train_generators_FAST(clients_dict, label_space_meta, dataset_meta, device, logger, 
                          use_new_gen_method=True, client_label_mask_dict=None, student_model=None):
    
    method_name = "New Gen Method" if use_new_gen_method else "Old Gen Method"
    logger.log(f"[Testing] Training Per-Dataset Generators Offline ({method_name}) using FAST...")

    generators_dict = {}
    FAST_INNER_STEPS = 5
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

        meta_optimizer = torch.optim.Adam(generator.parameters(), lr=fast_weight['lr'] * FAST_INNER_STEPS, betas=[0.5, 0.999])
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

        for epoch in tqdm(range(fast_weight['gen_epochs']), colour='green', ncols=100, desc=f"FAST-Gen:{ls_id}"):
            meta_optimizer.zero_grad() 
            total_epoch_loss = 0.0
            epoch_best_images = []

            if current_student is not None:
                current_student.eval()

            for client_idx in valid_client_indices:
                client_model = dataset_clients[client_idx]
                client_mask = current_mask[client_idx]
                
                valid_labels = torch.where(client_mask == 1)[0]
                if len(valid_labels) == 0: continue

                z_noise = torch.randn(batch_size, GEN_CONFIG['feat_gen_noise_dim'], device=device).requires_grad_()
                batch_labels = valid_labels[torch.randint(0, len(valid_labels), (batch_size,)).to(device)]
                
                fast_generator = ConditionalImageGenerator(num_local_classes, GEN_CONFIG['feat_gen_noise_dim'], 3, 32).to(device)
                fast_generator.load_state_dict(generator.state_dict())

                inner_optimizer = torch.optim.Adam([
                    {'params': fast_generator.parameters()},
                    {'params': [z_noise], 'lr': 0.01}
                ], lr=fast_weight['lr'], betas=[0.5, 0.999])

                best_cost = 1e6
                best_inputs_for_client = None

                for it in range(FAST_INNER_STEPS):
                    gen_imgs = fast_generator(z_noise, batch_labels)
                    #gen_imgs = jitter_and_flip(gen_imgs, lim=1./8., do_flip=True)
                    
                    _, logits_t = client_model(gen_imgs)
                    cls_loss = F.cross_entropy(logits_t, batch_labels)
                    
                    bn_loss = 0.0
                    client_hooks = all_client_bn_hooks[client_idx]
                    if len(client_hooks) > 0 and fast_weight['bn'] != 0:
                        bn_loss = sum([h.r_feature for h in client_hooks]) / len(client_hooks)

                    client_loss = (fast_weight['ce'] * cls_loss) + \
                                (fast_weight['bn'] * bn_loss) 
                    
                    if current_student is not None:
                        _, logits_s = current_student(gen_imgs)
                        mask = (logits_s.max(1)[1] == logits_t.max(1)[1]).float()
                        loss_adv = -(KLDiv(logits_s, logits_t.detach(), reduction='none').sum(1) * mask).mean()
                        client_loss += fast_weight['adv'] * loss_adv
                    
                    inner_optimizer.zero_grad()
                    client_loss.backward()

                    if client_loss.item() < best_cost:
                        best_cost = client_loss.item()
                        best_inputs_for_client = gen_imgs.detach()

                    if is_maml:
                        if it == 0: meta_optimizer.zero_grad()
                        fomaml_grad(generator, fast_generator)
                        if it == (FAST_INNER_STEPS - 1): meta_optimizer.step()

                    inner_optimizer.step()
                    if client_loss.item() < best_cost:
                        best_cost = client_loss.item()
                
                if best_inputs_for_client is not None:
                    epoch_best_images.append((client_idx, best_inputs_for_client))

                if not is_maml:
                    meta_optimizer.zero_grad()
                    reptile_grad(generator, fast_generator)
                    meta_optimizer.step()

                total_epoch_loss += best_cost

            for c_idx, b_imgs in epoch_best_images:
                image_pool.append((c_idx, b_imgs))
            
            if valid_client_count > 0:
                epoch_loss_tracker.append(total_epoch_loss / valid_client_count)

            if current_student is not None and len(image_pool) > 0:
                current_student.train()
                generator.eval() 
                
                for _ in range(GEN_CONFIG['kd_steps']):
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
            logger.log(f"     Dataset [{ls_id}] Generator | Epochs: {fast_weight['gen_epochs']} | Final Loss: {avg_loss:.4f}")

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
    'gen_epochs': 1500, 
    'lr': GEN_CONFIG.get('gen_lr', 5e-4),
    'ce': 0.5,      
    'bn': 10.0,      
    'adv': 1.0, 
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

        logger.log(f"     -> Loading CLIP ViT-B/32 label embeddings for {num_local_classes} classes...")
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

        #current_student = None

        for epoch in tqdm(range(nayer_weight['gen_epochs']), colour='magenta', ncols=100, desc=f"NAYER-Gen:{ls_id}"):
            gen_optimizer.zero_grad() 

            total_epoch_loss = 0.0
            total_adv_loss = 0.0

            if current_student is not None:
                current_student.eval()

            epoch_images = []

            generator.re_init_le()

            for client_idx in valid_client_indices:
                client_model = dataset_clients[client_idx]
                client_mask = current_mask[client_idx]
                
                valid_labels = torch.where(client_mask == 1)[0]
                if len(valid_labels) == 0: continue
                
                batch_labels = valid_labels[torch.randint(0, len(valid_labels), (batch_size,)).to(device)]
                
                gen_imgs = generator(targets=batch_labels)
                epoch_images.append((client_idx, gen_imgs.detach()))

                #gen_imgs = jitter_and_flip(gen_imgs, lim=1./8., do_flip=True)
                
                _, logits_t = client_model(gen_imgs)
                cls_loss = F.cross_entropy(logits_t, batch_labels)
                
                bn_loss = 0.0
                client_hooks = all_client_bn_hooks[client_idx]
                if len(client_hooks) > 0 and nayer_weight['bn'] != 0:
                    bn_loss = sum([h.r_feature for h in client_hooks]) / len(client_hooks)

                client_loss = (nayer_weight['ce'] * cls_loss) + \
                              (nayer_weight['bn'] * bn_loss)
                
                if current_student is not None and nayer_weight['adv'] > 0:
                    _, logits_s = current_student(gen_imgs)
                    mask = (logits_s.max(1)[1] == logits_t.max(1)[1]).float()
                    kl_val = F.kl_div(F.log_softmax(logits_s, dim=1), F.softmax(logits_t.detach(), dim=1), reduction='none').sum(1)
                    loss_adv = -(kl_val * mask).mean()

                    client_loss += nayer_weight['adv'] * loss_adv
                    total_adv_loss += loss_adv.item()
                
                (client_loss / valid_client_count).backward()
                total_epoch_loss += client_loss.item()

            for c_idx, imgs in epoch_images:
                image_pool.append((c_idx, imgs))
            
            if valid_client_count > 0:
                gen_optimizer.step()
                epoch_loss_tracker.append(total_epoch_loss / valid_client_count)

            if current_student is not None and len(image_pool) > 0:
                current_student.train()
                generator.eval() 
                
                for _ in range(GEN_CONFIG.get('kd_steps', 1)):
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
            logger.log(f"     Dataset [{ls_id}] Generator | Epochs: {nayer_weight['gen_epochs']} | Final Loss: {avg_loss:.4f}")

        for hooks in all_client_bn_hooks:
            for h in hooks:
                if hasattr(h, 'close'): h.close()
                elif hasattr(h, 'remove'): h.remove()

        generators_dict[ls_id] = generator

        # Draw
        draw_name = '_gen_img_new' if use_new_gen_method else '_gen_img_old'
        draw_nayer_samples('NAYER', draw_name, generator, class_names, logger, ls_id, device)

    return generators_dict


def draw_synthetic_samples(method, fname, generator, class_names, logger, ls_id, device):
    generator.eval()

    with torch.no_grad():
        save_dir = logger.log_dir
        num_samples_per_class = 1 
        num_local_classes = len(class_names)
        num_classes_to_plot = num_local_classes
        sample_labels = torch.arange(num_classes_to_plot).repeat(num_samples_per_class).to(device)
        sample_z = torch.randn(num_samples_per_class * num_classes_to_plot, GEN_CONFIG['feat_gen_noise_dim']).to(device)
        
        sample_imgs = generator(sample_z, sample_labels)
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