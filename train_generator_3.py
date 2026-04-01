import argparse
import os
import torch
import math
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np

from data.datasets import get_raw_dataset_transform, get_readable_class_names
from utils.logger import Logger
from utils.nets import (
    ConditionalImageGenerator, 
    MLP, CNN, ResNet, BasicBlock, 
    MobileNetV2, MobileNetV3, 
    LeNet, AlexNet, 
    ShuffleNetV2, SqueezeNet     
)
from utils.loss import (
    Gen_DiversityLoss, 
    total_variation_loss, 
    BNSM_Hook, 
    get_bn_loss,
    CAM_Hook,           
    get_gaussian_mask,  
    get_cam_loss       
)

def register_mapping(local_id_to_global_id, d1, l1, d2, l2):
    """
    將 (Dataset1, Label1) 與 (Dataset2, Label2) 連結到同一個 Global ID。
    """
    gid = None
    
    # Check d1:l1
    if d1 in local_id_to_global_id and l1 in local_id_to_global_id[d1]:
            gid = local_id_to_global_id[d1][l1]
    
    # Check d2:l2
    if gid is None and d2 in local_id_to_global_id and l2 in local_id_to_global_id[d2]:
            gid = local_id_to_global_id[d2][l2]
            
    # 如果兩邊都沒有，分配一個新的 GID
    if gid is None:
            current_max = -1
            for d in local_id_to_global_id:
                if local_id_to_global_id[d]:
                    current_max = max(current_max, max(local_id_to_global_id[d].values()))
            gid = current_max + 1
            
    if d1 not in local_id_to_global_id: local_id_to_global_id[d1] = {}
    local_id_to_global_id[d1][l1] = gid
    
    if d2 not in local_id_to_global_id: local_id_to_global_id[d2] = {}
    local_id_to_global_id[d2][l2] = gid


def test_discover_mappings_with_real_images(local_id_to_global_id, clients_dict, test_loaders_dict, label_space_meta, device, logger):
    logger.log("\n[Testing] Testing Cross-Dataset Label Mappings with REAL images ...")

    dataset_ids = list(clients_dict.keys())
    # entropy_threshold = 1.5
    tolerance_ratio = 0.65
    logger.log(f"tolerance_ratio: {tolerance_ratio}")

    def get_real_images(loader, target_label, num_samples=128):
        images_collected = []
        for imgs, labels in loader:
            mask = (labels == target_label)
            valid_imgs = imgs[mask]
            images_collected.append(valid_imgs)
            if sum(x.size(0) for x in images_collected) >= num_samples:
                break
        
        if len(images_collected) == 0:
            return None
        
        images_cat = torch.cat(images_collected, dim=0)[:num_samples]
        return images_cat.to(device)

    evaluation_cache = {}
    attempted_records = {}

    def evaluate_mapping(src_id, label_idx, tgt_id):
        cache_key = (src_id, label_idx, tgt_id)
        
        if cache_key in evaluation_cache:
            return evaluation_cache[cache_key]

        tgt_clients = clients_dict.get(tgt_id, [])
        tgt_names = label_space_meta[tgt_id]

        max_ent_tgt = math.log(len(tgt_names))
        dynamic_thresh_tgt = max_ent_tgt * tolerance_ratio

        src_loader = test_loaders_dict[src_id] 

        img_src_real = get_real_images(src_loader, label_idx, num_samples=128)
        if img_src_real is None:
            return None

        with torch.no_grad():
            all_probs = []
            valid_probs_tgt = []

            for c_model in tgt_clients:
                c_model.eval()
                _, logits = c_model(img_src_real)
                probs = F.softmax(logits, dim=1).mean(dim=0)
                ent = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                
                all_probs.append(probs)
                # if ent <= entropy_threshold:
                if ent <= dynamic_thresh_tgt:
                    valid_probs_tgt.append(probs)

            if len(valid_probs_tgt) == 0:
                avg_probs_tgt = torch.stack(all_probs).mean(dim=0)
                passed_threshold = False
            else:
                avg_probs_tgt = torch.stack(valid_probs_tgt).mean(dim=0)
                entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()
                # passed_threshold = (entropy_tgt <= entropy_threshold)
                passed_threshold = (entropy_tgt <= dynamic_thresh_tgt)
            
            entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()
            pred_tgt_idx = torch.argmax(avg_probs_tgt).item()
            pred_tgt_name = tgt_names[pred_tgt_idx]
            
            result = (pred_tgt_idx, pred_tgt_name, entropy_tgt, len(valid_probs_tgt), len(tgt_clients), passed_threshold)
            evaluation_cache[cache_key] = result
            return result

    global_table_records = {}

    def assign_and_get_gid(d1, l1, d2, l2):
        register_mapping(local_id_to_global_id, d1, l1, d2, l2)
        return local_id_to_global_id[d1][l1]
    
    # =====================================================================
    # ⭐ 這裡才是真正的守門員！用來記錄「已經走過並印在 Log 上的路徑」
    # =====================================================================
    processed_paths = set()

    for i, src_id in enumerate(dataset_ids):
        for j, tgt_id in enumerate(dataset_ids):
            if i == j: continue 

            src_names = label_space_meta[src_id]

            for label_idx, label_name in enumerate(src_names):
                
                # ⭐⭐⭐ 核心修復：如果這條路徑之前已經做過 Cycle-Back，直接無聲跳過！
                # 這樣 EMNIST 0 就不會再被拿去送給 MNIST 測一次！
                if (src_id, label_idx, tgt_id) in processed_paths:
                    continue

                if (src_id, label_idx) not in attempted_records:
                    attempted_records[(src_id, label_idx)] = []

                # ==========================================
                # 階段 1：正向預測 (A -> B)
                # ==========================================
                tgt_result = evaluate_mapping(src_id, label_idx, tgt_id)
                if tgt_result is None: continue 
                
                pred_tgt_idx, pred_tgt_name, ent_tgt, v_c, t_c, passed = tgt_result
                
                attempted_records[(src_id, label_idx)].append(f"->[{tgt_id}] '{pred_tgt_name}' (Ent:{ent_tgt:.2f})")

                # ⭐⭐⭐ 紀錄：正向路徑已處理
                processed_paths.add((src_id, label_idx, tgt_id))

                if not passed:
                    logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{ent_tgt:.2f} ⚠️ [Filtered: High Uncertainty]")
                    continue
                
                logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{ent_tgt:.2f} (Valid Clients: {v_c}/{t_c})")

                # ==========================================
                # 階段 2：Cycle-Back 反向驗證 (B -> A)
                # ==========================================
                src_result = evaluate_mapping(tgt_id, pred_tgt_idx, src_id)
                if src_result is None: continue

                pred_src_idx, pred_src_name, ent_src, v_c2, t_c2, passed2 = src_result

                # ⭐⭐⭐ 紀錄：反向路徑已處理 (這就是防止未來鬼打牆的關鍵！)
                processed_paths.add((tgt_id, pred_tgt_idx, src_id))

                if not passed2:
                    logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_name}' | ❓ | Entropy:{ent_src:.2f} ⚠️ [Filtered: Weak Cycle-Back]")
                    continue

                if pred_src_idx == label_idx:
                    logger.log(f"[Real Img Match] {src_id}:'{label_name}' <==> {tgt_id}:'{pred_tgt_name}' ✅ | Entropy:{ent_src:.2f} (Valid: {v_c2}/{t_c2})")
                    
                    gid = assign_and_get_gid(src_id, label_idx, tgt_id, pred_tgt_idx)
                    if gid not in global_table_records:
                        global_table_records[gid] = set()
                    
                    global_table_records[gid].add(f"[{src_id}] '{label_name}' -> [{tgt_id}] '{pred_tgt_name}' (Ent: {ent_tgt:.2f})")
                    global_table_records[gid].add(f"[{tgt_id}] '{pred_tgt_name}' -> [{src_id}] '{label_name}' (Ent: {ent_src:.2f})")
                else:
                    logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_name}' | ❌ | Entropy:{ent_src:.2f} ⚠️ [Filtered: Cycle-Back Mismatch]")

    # ==========================================
    # 補齊所有落單的類別 (分配全新 GID)
    # ==========================================
    for d_id in label_space_meta.keys():
        if d_id not in local_id_to_global_id:
            local_id_to_global_id[d_id] = {}
        
        class_names = label_space_meta[d_id]
        for l_id in range(len(class_names)):
            if l_id not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                
                local_id_to_global_id[d_id][l_id] = current_max + 1

    # ==========================================
    # 階段 3：印出對齊清單總表
    # ==========================================
    gid_to_classes = {}
    for d_id, map_dict in local_id_to_global_id.items():
        for l_id, gid in map_dict.items():
            cname = label_space_meta[d_id][l_id]
            if gid not in gid_to_classes:
                gid_to_classes[gid] = []
            gid_to_classes[gid].append((d_id, l_id, cname))

    logger.log("\n=========================================================================================================")
    logger.log(f"{'Global ID':<10} | {'Aligned Classes (Members)':<45} | {'Mapping Records & Entropies'}")
    logger.log("---------------------------------------------------------------------------------------------------------")
    for gid in sorted(gid_to_classes.keys()):
        members_str = ", ".join([f"[{m[0]}] '{m[2]}'" for m in gid_to_classes[gid]])
        
        if gid in global_table_records and len(global_table_records[gid]) > 0:
            records_str = " ; ".join(global_table_records[gid])
            logger.log(f"{gid:<10} | {members_str:<45} | {records_str}")
        else:
            orphaned_d_id, orphaned_l_id, _ = gid_to_classes[gid][0]
            attempts = attempted_records.get((orphaned_d_id, orphaned_l_id), [])
            if len(attempts) > 0:
                attempts_str = " ; ".join(attempts)
                logger.log(f"{gid:<10} | {members_str:<45} | ⚠️ No Match | Attempts: {attempts_str}")
            else:
                logger.log(f"{gid:<10} | {members_str:<45} | ⚠️ No Match (No Attempts Logged)")
    logger.log("=========================================================================================================\n")


def img_train_dataset_generators(clients_dict, label_space_meta, dataset_meta, device, logger):
    logger.log("[Testing] Training Per-Dataset Generators Offline ...")

    div_loss_fn = Gen_DiversityLoss(metric='l1').to(device)

    feat_gen_noise_dim = 128

    gen_epochs = 1500
    gen_lr = 5e-4

    alpha_ce = 1.0
    alpha_bn = 0.05

    alpha_cam = 0.0
    tv_weight = 0.005

    #target_mask = get_gaussian_mask(size=32, device=device, sigma=0.3)

    generators_dict = {}
    
    for ls_id, dataset_clients in clients_dict.items():
        if len(dataset_clients) == 0:
            continue

        class_names = label_space_meta[ls_id]
        num_local_classes = len(class_names)

        generator = ConditionalImageGenerator(
            num_classes=num_local_classes, 
            noise_dim=feat_gen_noise_dim,
            img_channels=3, 
            img_size=32   
        ).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr)
        
        bn_hooks = []
        cam_hooks = []

        for client_model in dataset_clients:
            client_model.eval() 
            for param in client_model.parameters():      
                param.requires_grad = False

            last_conv = None

            for module in client_model.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False

                if isinstance(module, torch.nn.BatchNorm2d):
                    bn_hooks.append(BNSM_Hook(module))

            #     if isinstance(module, torch.nn.Conv2d):
            #         last_conv = module

            # if last_conv is not None:
            #     cam_hooks.append(CAM_Hook(last_conv))

        generator.train()
        batch_size = 64
        epoch_loss_tracker = []

        for epoch in tqdm(range(gen_epochs), colour='blue', ncols=100, desc=f"Gen:{ls_id}"):
            class_order = torch.randperm(num_local_classes).tolist()

            #for c_idx in class_order:
            for _ in range(10):
                batch_labels = torch.randint(0, num_local_classes, (batch_size,)).to(device)
                z_noise = torch.randn(batch_size, feat_gen_noise_dim).to(device)

                optimizer.zero_grad()
                gen_imgs = generator(z_noise, batch_labels)
                gen_imgs.retain_grad()
                
                total_cls_loss = 0.0

                for client_model in dataset_clients:
                    
                    _, logits = client_model(gen_imgs)
                    
                    total_cls_loss += F.cross_entropy(logits, batch_labels)

                cls_loss = total_cls_loss / len(dataset_clients)
                tv_loss = total_variation_loss(gen_imgs)
                #cam_loss = get_cam_loss(cam_hooks, target_mask) if len(cam_hooks) > 0 else 0.0
                
                bn_loss = 0.0
                if len(bn_hooks) > 0:
                    try:
                        bn_loss = get_bn_loss(bn_hooks)
                    except Exception:
                        bn_loss = sum([h.r_feature for h in bn_hooks]) / len(bn_hooks)
                else:
                    bn_loss = 0.0 

                loss = (alpha_ce * cls_loss) + (alpha_bn * bn_loss) + (tv_weight * tv_loss)
                #loss = (alpha_ce * cls_loss) + (tv_weight * tv_loss) + (alpha_cam * cam_loss)
                loss.backward()
                optimizer.step()
                epoch_loss_tracker.append(loss.item())

        avg_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
        logger.log(f"     Dataset [{ls_id}] Generator | Epochs: {gen_epochs} | Final Loss: {avg_loss:.4f}")

        for h in bn_hooks + cam_hooks:
            if hasattr(h, 'close'):
                h.close()
            elif hasattr(h, 'remove'):
                h.remove()

        generators_dict[ls_id] = generator

        # Draw
        generator.eval()
        with torch.no_grad():
            os.makedirs('test_img', exist_ok=True) 
            save_dir = logger.log_dir

            num_samples_per_class = 1 
            num_classes_to_plot = num_local_classes
            sample_labels = torch.arange(num_classes_to_plot).repeat(num_samples_per_class).to(device)
            sample_z = torch.randn(num_samples_per_class * num_classes_to_plot, feat_gen_noise_dim).to(device)
            
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
            save_path = os.path.join(save_dir, f'{ls_id}_offline_generated_samples_2.png')
            plt.savefig(save_path)
            plt.close() 
            logger.log(f"     -> Offlline Sample images saved to {save_path}")

    logger.log("[Testing] Offline Per-Dataset Generators trained.")

    logger.log(f"gen_epochs: {gen_epochs}, gen_lr: {gen_lr}, alpha_ce: {alpha_ce}, alpha_bn: {alpha_bn}, tv_weight: {tv_weight}, alpha_cam: {alpha_cam}")

    return generators_dict

def img_discover_label_mappings(local_id_to_global_id, generators_dict, clients_dict, label_space_meta, device, logger):
    logger.log("[Testing] Discovering Cross-Dataset Label Mappings ...")
    dataset_ids = list(generators_dict.keys())

    feat_gen_noise_dim = 128
    tolerance_ratio = 0.65
    logger.log(f"tolerance_ratio: {tolerance_ratio}")

    evaluation_cache = {}
    
    attempted_records = {}

    def evaluate_mapping(src_id, label_idx, tgt_id):
        cache_key = (src_id, label_idx, tgt_id)
        
        if cache_key in evaluation_cache:
            return evaluation_cache[cache_key]

        tgt_clients = clients_dict.get(tgt_id, [])
        src_gen = generators_dict[src_id]
        tgt_names = label_space_meta[tgt_id]

        max_ent_tgt = math.log(len(tgt_names))
        dynamic_thresh_tgt = max_ent_tgt * tolerance_ratio

        src_gen.eval()
        num_samples = 128
        z = torch.randn(num_samples, feat_gen_noise_dim).to(device)
        label_tensor = torch.tensor([label_idx] * num_samples).to(device)

        with torch.no_grad():
            img_src = src_gen(z, label_tensor)
            
            all_probs = []
            valid_probs_tgt = []

            for c_model in tgt_clients:
                c_model.eval()
                _, logits = c_model(img_src)
                
                probs = F.softmax(logits, dim=1).mean(dim=0)
                ent = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                
                all_probs.append(probs)
                if ent <= dynamic_thresh_tgt:  
                    valid_probs_tgt.append(probs)

            if len(valid_probs_tgt) == 0:
                avg_probs_tgt = torch.stack(all_probs).mean(dim=0)
                passed_threshold = False
            else:
                avg_probs_tgt = torch.stack(valid_probs_tgt).mean(dim=0)
                entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()
                passed_threshold = (entropy_tgt <= dynamic_thresh_tgt)

            entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()
            pred_tgt_idx = torch.argmax(avg_probs_tgt).item()
            pred_tgt_name = tgt_names[pred_tgt_idx]

            result = (pred_tgt_idx, pred_tgt_name, entropy_tgt, len(valid_probs_tgt), len(tgt_clients), passed_threshold)
            evaluation_cache[cache_key] = result
            return result

    global_table_records = {}

    def assign_and_get_gid(d1, l1, d2, l2):
        register_mapping(local_id_to_global_id, d1, l1, d2, l2)
        return local_id_to_global_id[d1][l1]

    processed_paths = set()

    for i, src_id in enumerate(dataset_ids):
        for j, tgt_id in enumerate(dataset_ids):
            if i == j: continue 

            src_names = label_space_meta[src_id]

            for label_idx, label_name in enumerate(src_names):
                
                if (src_id, label_idx, tgt_id) in processed_paths:
                    continue

                if (src_id, label_idx) not in attempted_records:
                    attempted_records[(src_id, label_idx)] = []
                
                # ==========================================
                # 階段 1：正向預測 (A -> B)
                # ==========================================
                tgt_result = evaluate_mapping(src_id, label_idx, tgt_id)
                if tgt_result is None: continue 
                
                pred_tgt_idx, pred_tgt_name, ent_tgt, v_c, t_c, passed = tgt_result
                
                attempted_records[(src_id, label_idx)].append(f"->[{tgt_id}] '{pred_tgt_name}' (Ent:{ent_tgt:.2f})")

                processed_paths.add((src_id, label_idx, tgt_id))

                if not passed:
                    logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{ent_tgt:.2f} ⚠️ [Filtered: High Uncertainty]")
                    continue
                
                logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{ent_tgt:.2f} (Valid Clients: {v_c}/{t_c})")

                # ==========================================
                # 階段 2：Cycle-Back 反向驗證 (B -> A)
                # ==========================================
                src_result = evaluate_mapping(tgt_id, pred_tgt_idx, src_id)
                if src_result is None: continue

                pred_src_idx, pred_src_name, ent_src, v_c2, t_c2, passed2 = src_result

                processed_paths.add((tgt_id, pred_tgt_idx, src_id))

                if not passed2:
                    logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_name}' | ❓ | Entropy:{ent_src:.2f} ⚠️ [Filtered: Weak Cycle-Back]")
                    continue

                if pred_src_idx == label_idx:
                    logger.log(f"[Match] {src_id}:'{label_name}' <==> {tgt_id}:'{pred_tgt_name}' ✅ | Entropy:{ent_src:.2f} (Valid: {v_c2}/{t_c2})")
                    
                    gid = assign_and_get_gid(src_id, label_idx, tgt_id, pred_tgt_idx)
                    
                    if gid not in global_table_records:
                        global_table_records[gid] = set() 
                    
                    global_table_records[gid].add(f"[{src_id}] '{label_name}' -> [{tgt_id}] '{pred_tgt_name}' (Ent: {ent_tgt:.2f})")
                    global_table_records[gid].add(f"[{tgt_id}] '{pred_tgt_name}' -> [{src_id}] '{label_name}' (Ent: {ent_src:.2f})")
                else:
                    logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_name}' | ❌ | Entropy:{ent_src:.2f} ⚠️ [Filtered: Cycle-Back Mismatch]")

    # ==========================================
    # 補齊所有落單的類別 (分配全新 GID)
    # ==========================================
    for d_id in label_space_meta.keys():
        if d_id not in local_id_to_global_id:
            local_id_to_global_id[d_id] = {}
        
        class_names = label_space_meta[d_id]
        for l_id in range(len(class_names)):
            if l_id not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                
                local_id_to_global_id[d_id][l_id] = current_max + 1

    # ==========================================
    # 階段 3：印出對齊清單總表
    # ==========================================
    gid_to_classes = {}
    for d_id, map_dict in local_id_to_global_id.items():
        for l_id, gid in map_dict.items():
            cname = label_space_meta[d_id][l_id]
            if gid not in gid_to_classes:
                gid_to_classes[gid] = []
            # 存成 tuple 方便後面抓出 d_id, l_id 去查 attempted_records
            gid_to_classes[gid].append((d_id, l_id, cname))

    logger.log("\n=========================================================================================================")
    logger.log(f"{'Global ID':<10} | {'Aligned Classes (Members)':<45} | {'Mapping Records & Entropies'}")
    logger.log("---------------------------------------------------------------------------------------------------------")
    for gid in sorted(gid_to_classes.keys()):
        # 把 members 轉成字串
        members_str = ", ".join([f"[{m[0]}] '{m[2]}'" for m in gid_to_classes[gid]])
        
        if gid in global_table_records and len(global_table_records[gid]) > 0:
            records_str = " ; ".join(global_table_records[gid])
            logger.log(f"{gid:<10} | {members_str:<45} | {records_str}")
        else:
            # 這是落單的類別！
            orphaned_d_id, orphaned_l_id, _ = gid_to_classes[gid][0]
            
            # 從我們剛剛存的 attempted_records 把它送去別人的結果拉出來！
            attempts = attempted_records.get((orphaned_d_id, orphaned_l_id), [])
            if len(attempts) > 0:
                attempts_str = " ; ".join(attempts)
                logger.log(f"{gid:<10} | {members_str:<45} | ⚠️ No Match | Attempts: {attempts_str}")
            else:
                logger.log(f"{gid:<10} | {members_str:<45} | ⚠️ No Match (No Attempts Logged)")
    logger.log("=========================================================================================================\n")


def set_seed(seed):
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)                
        torch.backends.cudnn.deterministic = True        
        torch.backends.cudnn.benchmark = False 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing .pth files")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    logger = Logger(args)
    logger.log(f"Loading checkpoints from {args.model_path}")

    set_seed(args.seed)

    target_datasets = ['MNIST', 'EMNIST', 'CIFAR10'] 
    
    DATASET_META = { 
        #'EMNIST': {'in_ch': 3, 'classes': 47,  'size': 32},
        'EMNIST': {'in_ch': 3, 'classes': 62,  'size': 32},
        'MNIST':  {'in_ch': 3, 'classes': 10,  'size': 32}, 
        'CIFAR10':  {'in_ch': 3, 'classes': 10,  'size': 32}, 
    }

    DATA_ROOT = './data/raw'

    label_space_meta = {}
    for d_name in target_datasets:
        label_space_meta[d_name] = get_readable_class_names(d_name, root=DATA_ROOT)

    clients_dict = {}

    for d_name in target_datasets:
        clients_dict[d_name] = [] # 改成 List，用來裝這一個 dataset 旗下的所有異構模型
        meta = DATASET_META[d_name]

        if not os.path.exists(args.model_path):
            logger.log(f"Path {args.model_path} doesn't exist.")
            continue

        for filename in os.listdir(args.model_path):
            # 檔名格式為 "client_model_CIFAR10_c22_MLP.pth" 這種類型
            # 因此只要確保檔名以 client_model_{d_name} 開頭即可
            if filename.startswith(f"client_model_{d_name}_c") and filename.endswith(".pth"):
                ckpt_path = os.path.join(args.model_path, filename)
                
                # 從檔名解析出架構名稱 (例如從 "client_model_CIFAR10_c22_MLP.pth" 去掉 ".pth" 後，以 "_" 切割取最後一部分)
                arch_name = filename.replace(".pth", "").split("_")[-1]

                # 根據檔名分配不同的架構
                if arch_name == 'MLP':
                    # 注意：MLP 需要傳入 img_size
                    model = MLP(in_channels=meta['in_ch'], num_classes=meta['classes'], img_size=meta['size']).to(args.device)
                elif arch_name == 'CNN':
                    model = CNN(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ResNet8':
                    model = ResNet(BasicBlock, [1, 1, 1, 0], in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ResNet18':
                    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'MobileNetV2':
                    model = MobileNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'MobileNetV3':
                    model = MobileNetV3(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'LeNet':
                    model = LeNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'AlexNet':
                    model = AlexNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ShuffleNet':
                    model = ShuffleNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'SqueezeNet':
                    model = SqueezeNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                else:
                    logger.log(f"Warning: Unknown model architecture '{arch_name}' in {filename}. Skipping.")
                    continue

                # 載入權重並加入 Dict
                model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
                model.eval()
                clients_dict[d_name].append(model)
                logger.log(f"Loaded {arch_name} for {d_name} from {filename}")

        if len(clients_dict[d_name]) == 0:
            logger.log(f"WARNING: No models loaded for dataset: {d_name}.")


    test_loaders_dict = {}
    for d_name in target_datasets:
        test_dataset = get_raw_dataset_transform(name=d_name, root=DATA_ROOT, train=False)
        test_loaders_dict[d_name] = DataLoader(test_dataset, batch_size=128, shuffle=False)

    valid_datasets = [d for d, models in clients_dict.items() if len(models) > 0]
    
    if len(valid_datasets) > 1:
        local_id_to_global_id = {}

        # test_discover_mappings_with_real_images(local_id_to_global_id, clients_dict, test_loaders_dict, label_space_meta, args.device, logger)
    
        generators_dict = img_train_dataset_generators(clients_dict, label_space_meta, DATASET_META, args.device, logger)
        
        img_discover_label_mappings(local_id_to_global_id, generators_dict, clients_dict, label_space_meta, args.device, logger)
    else:
        logger.log("Not enough models loaded to perform cross-dataset testing.")

if __name__ == "__main__":
    main()