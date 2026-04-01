import torch
import torch.nn.functional as F
import math
import os
import csv

img_num_samples = 64
feat_gen_noise_dim = 128


def get_gen_images(dataset_id, label_idx, args=None, gen_dict=None):
    gen = gen_dict[dataset_id]

    if isinstance(gen, dict):
        pool = gen.get(label_idx)
        if pool is None or len(pool) == 0:
            return torch.randn(img_num_samples, 3, 32, 32).to(args.device)
        
        indices = torch.randint(0, len(pool), (img_num_samples,))
        return pool[indices].to(args.device)
    
    gen.eval()

    z = torch.randn(img_num_samples, feat_gen_noise_dim).to(args.device)
    y = torch.tensor([label_idx] * img_num_samples).to(args.device)

    with torch.no_grad():
        if type(gen).__name__ == 'NLGenerator':
            gen.re_init_le()  
            imgs = gen(targets=y)
        else:
            imgs = gen(z, y)

    return imgs

def get_real_images(dataset_id, label_idx, args=None, test_loaders=None):
    loader = test_loaders[dataset_id]
    images_collected = []
    for imgs, labels in loader:
        mask = (labels == label_idx)
        valid_imgs = imgs[mask]
        images_collected.append(valid_imgs)
        if sum(x.size(0) for x in images_collected) >= img_num_samples:
            break
    
    if len(images_collected) == 0:
        return None
    
    images_cat = torch.cat(images_collected, dim=0)[:img_num_samples]
    return images_cat.to(args.device)


def register_mapping(local_id_to_global_id, d1, l1, d2, l2):
    """
    將 (Dataset1, Label1) 和 (Dataset2, Label2) 連到同一個 Global ID
    """
    gid = None
    
    # Check d1:l1
    if d1 in local_id_to_global_id and l1 in local_id_to_global_id[d1]:
            gid = local_id_to_global_id[d1][l1]
    
    # Check d2:l2
    if gid is None and d2 in local_id_to_global_id and l2 in local_id_to_global_id[d2]:
            gid = local_id_to_global_id[d2][l2]
            
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


def evaluate_mapping_results(dataset_ids, label_space_meta, local_id_to_global_id):
    TP = FP = TN = FN = 0

    for i in range(len(dataset_ids)):
        for j in range(i + 1, len(dataset_ids)):
            d1 = dataset_ids[i]
            d2 = dataset_ids[j]
            
            for l1_idx, l1_name in enumerate(label_space_meta[d1]):
                for l2_idx, l2_name in enumerate(label_space_meta[d2]):
                    is_ground_truth_match = (str(l1_name).strip().lower() == str(l2_name).strip().lower())
                    
                    gid1 = local_id_to_global_id.get(d1, {}).get(l1_idx)
                    gid2 = local_id_to_global_id.get(d2, {}).get(l2_idx)
                    
                    is_predicted_match = (gid1 is not None) and (gid2 is not None) and (gid1 == gid2)

                    if is_ground_truth_match and is_predicted_match: 
                        TP += 1
                    elif not is_ground_truth_match and is_predicted_match: 
                        FP += 1
                    elif not is_ground_truth_match and not is_predicted_match: 
                        TN += 1
                    elif is_ground_truth_match and not is_predicted_match: 
                        FN += 1
    
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    average_accuracy = (recall + specificity) / 2.0
    
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Recall": recall, "Specificity": specificity, "AvgAccuracy": average_accuracy
    }


def save_mapping_results_to_csv(save_dir, method_name, filename, metrics, extra_info_header, extra_info_values):
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = os.path.join(save_dir, method_name + '_' + filename)
    file_exists = os.path.isfile(csv_filename)
    
    base_header = ['recall', 'specificity', 'average_accuracy', 'TP', 'FP', 'TN', 'FN']
    base_values = [
        metrics['Recall'], metrics['Specificity'], metrics['AvgAccuracy'],
        metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
    ]
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(extra_info_header + base_header)
        writer.writerow(extra_info_values + base_values)


def old_entropy_filter(tgt_clients, images, tgt_names, entropy_ratio):
    """ 
    [Old Method]
    """
    if not tgt_clients: return None

    entropy_thresh = math.log(len(tgt_names)) * entropy_ratio

    all_probs = []
    with torch.no_grad():
        for model in tgt_clients:
            model.eval()
            _, logits = model(images)
            probs = F.softmax(logits, dim=1).mean(dim=0)
            all_probs.append(probs) 
    
    if not all_probs: 
        return None

    avg_probs = torch.stack(all_probs).mean(dim=0)
    final_ent = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8)).item()

    if final_ent > entropy_thresh:
        return None 
    
    pred_idx = torch.argmax(avg_probs).item()
    return {
        'pred_idx': pred_idx,
        'pred_name': tgt_names[pred_idx],
        'entropy': final_ent,
        'valid_count': len(all_probs), 
        'total_count': len(tgt_clients)
    }

def new_entropy_filter(tgt_clients, images, tgt_names, entropy_ratio):
    """
    [New Method]
    """
    if not tgt_clients: return None

    entropy_thresh = math.log(len(tgt_names)) * entropy_ratio

    valid_probs = []
    with torch.no_grad():
        for model in tgt_clients:
            model.eval()
            _, logits = model(images)
            probs = F.softmax(logits, dim=1).mean(dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            if entropy <= entropy_thresh:
                valid_probs.append(probs)
    
    if not valid_probs: 
        return None

    avg_probs = torch.stack(valid_probs).mean(dim=0)
    final_ent = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8)).item()

    if final_ent > entropy_thresh: 
        return None 

    pred_idx = torch.argmax(avg_probs).item()
    return {
        'pred_idx': pred_idx,
        'pred_name': tgt_names[pred_idx],
        'entropy': final_ent,
        'valid_count': len(valid_probs),
        'total_count': len(tgt_clients)
    }


def label_mapping(get_images_func, dataset_ids, clients_dict, label_space_meta, entropy_ratio, use_new_entropy_method, logger, **get_images_kwargs):
    """
    """
    local_id_to_global_id = {}
    
    logger.log(f"dynamic_entropy_ratio: {entropy_ratio}")

    prediction_method = new_entropy_filter if use_new_entropy_method else old_entropy_filter
    
    processed_pairs = set()

    for d_id in dataset_ids:
        local_id_to_global_id[d_id] = {}

    for src_id in dataset_ids:
        for tgt_id in dataset_ids:
            if src_id == tgt_id: continue
            
            src_names = label_space_meta[src_id]
            for l_idx, l_name in enumerate(src_names):
                if (src_id, l_idx, tgt_id) in processed_pairs: continue

                # 1. Get Source Images
                if get_images_kwargs:
                    imgs_src = get_images_func(src_id, l_idx, **get_images_kwargs)
                else:
                    imgs_src = get_images_func(src_id, l_idx)

                if imgs_src is None: continue

                # 2. Forward Predict (Src -> Tgt)
                tgt_names = label_space_meta[tgt_id]
                tgt_clients = clients_dict.get(tgt_id, [])
                
                res_tgt = prediction_method(tgt_clients, imgs_src, tgt_names, entropy_ratio)
                processed_pairs.add((src_id, l_idx, tgt_id))

                if res_tgt is None: 
                    logger.log(f"{src_id}:'{l_name}' -> {tgt_id} ❌ (Uncertain)")
                    continue
                
                pred_tgt_idx = res_tgt['pred_idx']
                pred_tgt_name = res_tgt['pred_name']
                logger.log(f"{src_id}:'{l_name}' -> {tgt_id} | Predict: '{pred_tgt_name}' (Ent: {res_tgt['entropy']:.2f})")

                if get_images_kwargs:
                    imgs_tgt = get_images_func(tgt_id, pred_tgt_idx, **get_images_kwargs)
                else:
                    imgs_tgt = get_images_func(tgt_id, pred_tgt_idx)
                  
                if imgs_tgt is None: continue

                src_clients_for_back = clients_dict.get(src_id, []) 
                src_names_for_back = label_space_meta[src_id]

                res_back = prediction_method(src_clients_for_back, imgs_tgt, src_names_for_back, entropy_ratio)
                processed_pairs.add((tgt_id, pred_tgt_idx, src_id))

                if res_back is None: continue

                if res_back is None:
                    logger.log(f"  -> Cycle Check: {tgt_id} -> {src_id} ❌ (Back-check Uncertain)")
                    continue
                
                if res_back['pred_idx'] == l_idx:
                    logger.log(f"  ✅ [Match] {src_id}:'{l_name}' <==> {tgt_id}:'{pred_tgt_name}' (Ent: {res_back['entropy']:.2f})")
                    
                    register_mapping(local_id_to_global_id, src_id, l_idx, tgt_id, pred_tgt_idx)
                else:
                    logger.log(f"  ⚠️ [Mismatch] Cycle Back predicted '{res_back['pred_name']}' (Expected '{l_name}') (Ent: {res_back['entropy']:.2f})")

    for d_id in dataset_ids:
        for l_idx in range(len(label_space_meta[d_id])):
            if l_idx not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                local_id_to_global_id[d_id][l_idx] = current_max + 1
                
    return local_id_to_global_id






