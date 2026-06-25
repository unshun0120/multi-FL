import torch
import torch.nn.functional as F
import math
import os
import csv

img_num_samples = 8
feat_gen_noise_dim = 128

gan_image_cache = {}
ddpm_image_cache = {}
real_image_cache = {}

def clear_image_caches():
    global ddpm_image_cache
    ddpm_image_cache.clear()
    real_image_cache.clear()
    gan_image_cache.clear()

def get_gen_images(dataset_id, label_idx, args=None, gen_dict=None):
    global ddpm_image_cache

    gen = gen_dict[dataset_id]

    if isinstance(gen, dict):
        pool = gen.get(label_idx)
        if pool is None or len(pool) == 0:
            return torch.randn(img_num_samples, 3, 32, 32).to(args.device)
        
        indices = torch.randint(0, len(pool), (img_num_samples,))
        return pool[indices].to(args.device)
    
    is_ddpm = type(gen).__name__ == 'DDPM'
    if is_ddpm and (dataset_id, label_idx) in ddpm_image_cache:
        return ddpm_image_cache[(dataset_id, label_idx)]
    elif not is_ddpm and (dataset_id, label_idx) in gan_image_cache:
        return gan_image_cache[(dataset_id, label_idx)]
    
    gen.eval()

    z = torch.randn(img_num_samples, feat_gen_noise_dim).to(args.device)
    y = torch.tensor([label_idx] * img_num_samples).to(args.device)

    with torch.no_grad():
        if type(gen).__name__ == 'NLGenerator':
            #gen.re_init_le()  
            imgs = gen(targets=y)
        elif type(gen).__name__ == 'DDPM':
            imgs, _ = gen.sample(img_num_samples, size=(3, 32, 32), device=args.device, guide_w=1.5, label=label_idx)
            ddpm_image_cache[(dataset_id, label_idx)] = imgs
        elif type(gen).__name__ == 'DDIM' :
            imgs, _ = gen.sample(img_num_samples, size=(3, 32, 32), device=args.device, guide_w=3.0, label=label_idx)
            ddpm_image_cache[(dataset_id, label_idx)] = imgs
        else:
            z = torch.randn(img_num_samples, feat_gen_noise_dim).to(args.device)
            imgs = gen(z, y)
            gan_image_cache[(dataset_id, label_idx)] = imgs

    return imgs

def get_real_images(dataset_id, label_idx, args=None, test_loaders=None):
    global real_image_cache

    cache_key = (dataset_id, label_idx)
    if cache_key in real_image_cache:
        return real_image_cache[cache_key].to(args.device)
    
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
    images_cat = images_cat.to(args.device)
    real_image_cache[cache_key] = images_cat

    return images_cat


def register_mapping(local_id_to_global_id, d1, l1, d2, l2):
    """
    將 (Dataset1, Label1) 和 (Dataset2, Label2) 連到同一個 Global ID
    """
    gid = None

    gid_1 = local_id_to_global_id.get(d1, {}).get(l1)
    gid_2 = local_id_to_global_id.get(d2, {}).get(l2)
    
    if gid_1 is None and gid_2 is None:
        current_max = -1
        for d in local_id_to_global_id:
            if local_id_to_global_id[d]:
                current_max = max(current_max, max(local_id_to_global_id[d].values()))
        new_gid = current_max + 1
    else:
        new_gid = gid_1 if gid_1 is not None else gid_2

    if d1 in local_id_to_global_id:
        for existing_label, existing_gid in local_id_to_global_id[d1].items():
            if existing_gid == new_gid and existing_label != l1:
                #print(f"⚠️ Conflict in {d1}: Label '{existing_label}' and '{l1}' both map to Global ID {gid}. Discarding mapping for '{l1}'.")
                return

    if d2 in local_id_to_global_id:
        for existing_label, existing_gid in local_id_to_global_id[d2].items():
            if existing_gid == new_gid and existing_label != l2:
                #print(f"⚠️ Conflict in {d2}: Label '{existing_label}' and '{l2}' both map to Global ID {gid}. Discarding mapping for '{l2}'.")
                return
            
    if d1 not in local_id_to_global_id: local_id_to_global_id[d1] = {}
    if d2 not in local_id_to_global_id: local_id_to_global_id[d2] = {}

    if gid_1 is not None and gid_2 is not None and gid_1 != gid_2:
        clients_in_g1 = [d_name for d_name, lbls in local_id_to_global_id.items() if gid_1 in lbls.values()]
        clients_in_g2 = [d_name for d_name, lbls in local_id_to_global_id.items() if gid_2 in lbls.values()]
        
        if set(clients_in_g1).intersection(set(clients_in_g2)):
            return  
            
        for d_name in local_id_to_global_id:
            for label_name, gid in list(local_id_to_global_id[d_name].items()): 
                if gid == gid_2:
                    local_id_to_global_id[d_name][label_name] = gid_1
        new_gid = gid_1

    local_id_to_global_id[d1][l1] = new_gid
    local_id_to_global_id[d2][l2] = new_gid


def evaluate_mapping_results(dataset_ids, label_space_meta, local_id_to_global_id, valid_labels_dict=None):
    TP = FP = TN = FN = 0

    for i in range(len(dataset_ids)):
        for j in range(i + 1, len(dataset_ids)):
            d1 = dataset_ids[i]
            d2 = dataset_ids[j]
            
            for l1_idx, l1_name in enumerate(label_space_meta[d1]):
                if valid_labels_dict is not None and d1 in valid_labels_dict:
                    if l1_idx not in valid_labels_dict[d1]:
                        continue

                for l2_idx, l2_name in enumerate(label_space_meta[d2]):
                    if valid_labels_dict is not None and d2 in valid_labels_dict:
                        if l2_idx not in valid_labels_dict[d2]:
                            continue

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
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    average_accuracy = (recall + specificity) / 2.0
    mcc_denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / mcc_denominator if mcc_denominator > 0 else 0.0
    
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Recall": recall, "Specificity": specificity, "Precision": precision,
        "F1-Score": f1_score, "AvgAccuracy": average_accuracy, "MCC": mcc
    }


def save_mapping_results_to_csv(save_dir, method_name, filename, metrics, extra_info_header, extra_info_values):
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = os.path.join(save_dir, method_name + '_' + filename)
    file_exists = os.path.isfile(csv_filename)
    
    base_header = ['recall', 'specificity', 'precision', 'average_accuracy', 'F1-Score', 'MCC', 'TP', 'FP', 'TN', 'FN']
    base_values = [
        metrics['Recall'], metrics['Specificity'], metrics['Precision'],
        metrics['AvgAccuracy'], metrics['F1-Score'], metrics['MCC'], metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
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


def label_mapping(get_images_func, dataset_ids, clients_dict, label_space_meta, entropy_ratio, use_new_entropy_method, logger, valid_labels_dict=None, **get_images_kwargs):
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
                
                if valid_labels_dict is not None and src_id in valid_labels_dict:
                    if l_idx not in valid_labels_dict[src_id]:
                        continue

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
                    #logger.log(f"{src_id}:'{l_name}' -> {tgt_id} ❌ (Uncertain)")
                    print(f"{src_id}:'{l_name}' -> {tgt_id} ❌ (Uncertain)")
                    continue
                
                pred_tgt_idx = res_tgt['pred_idx']
                pred_tgt_name = res_tgt['pred_name']
                if valid_labels_dict is not None and tgt_id in valid_labels_dict:
                    if pred_tgt_idx not in valid_labels_dict[tgt_id]:
                        continue
                #logger.log(f"{src_id}:'{l_name}' -> {tgt_id} | Predict: '{pred_tgt_name}' (Ent: {res_tgt['entropy']:.2f})")
                print(f"{src_id}:'{l_name}' -> {tgt_id} | Predict: '{pred_tgt_name}' (Ent: {res_tgt['entropy']:.2f})")

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
                    #logger.log(f"  -> Cycle Check: {tgt_id} -> {src_id} ❌ (Back-check Uncertain)")
                    print(f"  -> Cycle Check: {tgt_id} -> {src_id} ❌ (Back-check Uncertain)")
                    continue
                
                if res_back['pred_idx'] == l_idx:
                    #logger.log(f"  ✅ [Match] {src_id}:'{l_name}' <==> {tgt_id}:'{pred_tgt_name}' (Ent: {res_back['entropy']:.2f})")
                    print(f"  ✅ [Match] {src_id}:'{l_name}' <==> {tgt_id}:'{pred_tgt_name}' (Ent: {res_back['entropy']:.2f})")

                    register_mapping(local_id_to_global_id, src_id, l_idx, tgt_id, pred_tgt_idx)
                else:
                    #logger.log(f"  ⚠️ [Mismatch] Cycle Back predicted '{res_back['pred_name']}' (Expected '{l_name}') (Ent: {res_back['entropy']:.2f})")
                    print(f"  ⚠️ [Mismatch] Cycle Back predicted '{res_back['pred_name']}' (Expected '{l_name}') (Ent: {res_back['entropy']:.2f})")
                    
    for d_id in dataset_ids:
        for l_idx in range(len(label_space_meta[d_id])):
            if valid_labels_dict is not None and d_id in valid_labels_dict:
                if l_idx not in valid_labels_dict[d_id]:
                    continue

            if l_idx not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                local_id_to_global_id[d_id][l_idx] = current_max + 1
                
    return local_id_to_global_id


def single_direction_label_mapping(get_images_func, dataset_ids, clients_dict, label_space_meta, entropy_ratio, use_new_entropy_method, logger, valid_labels_dict=None, **get_images_kwargs):
    logger.log(f"Single-direction Mapping | dynamic_entropy_ratio: {entropy_ratio}")
    prediction_method = new_entropy_filter if use_new_entropy_method else old_entropy_filter
    
    mapped_edges = []
    
    for src_id in dataset_ids:
        for tgt_id in dataset_ids:
            if src_id == tgt_id: continue
            
            src_names = label_space_meta[src_id]
            for l_idx, l_name in enumerate(src_names):
                if valid_labels_dict is not None and src_id in valid_labels_dict:
                    if l_idx not in valid_labels_dict[src_id]: continue

                imgs_src = get_images_func(src_id, l_idx, **get_images_kwargs) if get_images_kwargs else get_images_func(src_id, l_idx)
                if imgs_src is None: continue

                tgt_names = label_space_meta[tgt_id]
                tgt_clients = clients_dict.get(tgt_id, [])
                
                res_tgt = prediction_method(tgt_clients, imgs_src, tgt_names, entropy_ratio)
                if res_tgt is not None:
                    pred_tgt_idx = res_tgt['pred_idx']
                    if valid_labels_dict is not None and tgt_id in valid_labels_dict:
                        if pred_tgt_idx not in valid_labels_dict[tgt_id]: continue
                    
                    mapped_edges.append((src_id, l_idx, tgt_id, pred_tgt_idx, res_tgt['entropy']))
                    print(f"  [Candidate] {src_id}:'{l_idx}' -> {tgt_id}:'{pred_tgt_idx}' (Ent: {res_tgt['entropy']:.2f})")

    local_id_to_global_id = {d_id: {} for d_id in dataset_ids}
    
    sorted_edges = sorted(mapped_edges, key=lambda x: x[4])
    
    for src_id, l_idx, tgt_id, pred_idx, ent in sorted_edges:
        register_mapping(local_id_to_global_id, src_id, l_idx, tgt_id, pred_idx)
        
    for d_id in dataset_ids:
        for l_idx in range(len(label_space_meta[d_id])):
            if valid_labels_dict is not None and d_id in valid_labels_dict and l_idx not in valid_labels_dict[d_id]:
                continue
            if l_idx not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                local_id_to_global_id[d_id][l_idx] = current_max + 1
                
    return local_id_to_global_id


def old_feature_entropy_filter(tgt_clients, features, tgt_names, entropy_ratio):
    """
    """
    if not tgt_clients or features is None: return None
    
    entropy_thresh = math.log(len(tgt_names)) * entropy_ratio
    all_probs = []
    
    with torch.no_grad():
        for model in tgt_clients:
            model.eval()
            logits = model.classifier(features)
            probs = F.softmax(logits, dim=1).mean(dim=0)
            all_probs.append(probs)
            
    if not all_probs: return None

    avg_probs = torch.stack(all_probs).mean(dim=0)
    final_ent = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8)).item()

    if final_ent > entropy_thresh: return None 

    pred_idx = torch.argmax(avg_probs).item()
    return {
        'pred_idx': pred_idx,
        'pred_name': tgt_names[pred_idx],
        'entropy': final_ent
    }

def new_feature_entropy_filter(tgt_clients, features, tgt_names, entropy_ratio):
    """
    """
    if not tgt_clients or features is None: return None
    
    entropy_thresh = math.log(len(tgt_names)) * entropy_ratio
    valid_probs = []
    
    with torch.no_grad():
        for model in tgt_clients:
            model.eval()
            logits = model.classifier(features)
            probs = F.softmax(logits, dim=1).mean(dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            if entropy <= entropy_thresh:
                valid_probs.append(probs)
                
    if not valid_probs: return None

    avg_probs = torch.stack(valid_probs).mean(dim=0)
    final_ent = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8)).item()

    if final_ent > entropy_thresh: return None 

    pred_idx = torch.argmax(avg_probs).item()
    return {
        'pred_idx': pred_idx,
        'pred_name': tgt_names[pred_idx],
        'entropy': final_ent
    }

def feature_bi_direction_label_mapping(dataset_features_dict, dataset_ids, clients_dict, label_space_meta, entropy_ratio, use_new_entropy_method, logger, valid_labels_dict=None):
    """
    """
    logger.log(f"Feature Bi-direction Mapping | dynamic_entropy_ratio: {entropy_ratio}")

    prediction_method = new_feature_entropy_filter if use_new_entropy_method else old_feature_entropy_filter
    local_id_to_global_id = {d_id: {} for d_id in dataset_ids}
    processed_pairs = set()

    for src_id in dataset_ids:
        for tgt_id in dataset_ids:
            if src_id == tgt_id: continue
            src_names = label_space_meta[src_id]
            
            for l_idx, l_name in enumerate(src_names):
                if valid_labels_dict is not None and src_id in valid_labels_dict and l_idx not in valid_labels_dict[src_id]:
                    continue
                if (src_id, l_idx, tgt_id) in processed_pairs: continue

                feats_src = dataset_features_dict.get(src_id, {}).get(l_idx)
                if feats_src is None: continue

                tgt_names = label_space_meta[tgt_id]
                tgt_clients = clients_dict.get(tgt_id, [])
                res_tgt = prediction_method(tgt_clients, feats_src, tgt_names, entropy_ratio)
                processed_pairs.add((src_id, l_idx, tgt_id))

                if res_tgt is None: 
                    #logger.log(f"{src_id}:'{l_name}' -> {tgt_id} ❌ (Uncertain)")
                    continue
                pred_tgt_idx = res_tgt['pred_idx']

                if valid_labels_dict is not None and tgt_id in valid_labels_dict and pred_tgt_idx not in valid_labels_dict[tgt_id]:
                    continue
                
                pred_tgt_name = res_tgt['pred_name']
                #logger.log(f"{src_id}:'{l_name}' -> {tgt_id} | Predict: '{pred_tgt_name}' (Ent: {res_tgt['entropy']:.2f})")

                feats_tgt = dataset_features_dict.get(tgt_id, {}).get(pred_tgt_idx)
                if feats_tgt is None: continue

                src_clients_for_back = clients_dict.get(src_id, [])
                src_names_for_back = label_space_meta[src_id]
                res_back = prediction_method(src_clients_for_back, feats_tgt, src_names_for_back, entropy_ratio)
                processed_pairs.add((tgt_id, pred_tgt_idx, src_id))

                if res_back is not None and res_back['pred_idx'] == l_idx:
                    #logger.log(f"  ✅ [Match] {src_id}:'{l_name}' <==> {tgt_id}:'{pred_tgt_name}' (Ent: {res_back['entropy']:.2f})")
                    register_mapping(local_id_to_global_id, src_id, l_idx, tgt_id, pred_tgt_idx)
                #else:
                    #logger.log(f"  ⚠️ [Mismatch] Cycle Back predicted '{res_back['pred_name']}' (Expected '{l_name}') (Ent: {res_back['entropy']:.2f})")
    
    for d_id in dataset_ids:
        for l_idx in range(len(label_space_meta[d_id])):
            if valid_labels_dict is not None and d_id in valid_labels_dict and l_idx not in valid_labels_dict[d_id]:
                continue
            if l_idx not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                local_id_to_global_id[d_id][l_idx] = current_max + 1
                
    return local_id_to_global_id


def image_cosine_similarity_mapping(get_images_func, dataset_ids, label_space_meta, cs_threshold, logger, valid_labels_dict=None, **get_images_kwargs):
    logger.log(f"Image Cosine Similarity Mapping | Cosine Similarity Threshold: {cs_threshold}")
    mapped_edges = []

    for src_id in dataset_ids:
        for tgt_id in dataset_ids:
            if src_id == tgt_id: continue

            src_names = label_space_meta[src_id]
            for l_idx, l_name in enumerate(src_names):
                if valid_labels_dict is not None and src_id in valid_labels_dict and l_idx not in valid_labels_dict[src_id]:
                    continue

                imgs_src = get_images_func(src_id, l_idx, **get_images_kwargs) if get_images_kwargs else get_images_func(src_id, l_idx)
                if imgs_src is None: continue

                proto_src = imgs_src.view(imgs_src.size(0), -1).mean(dim=0).unsqueeze(0)

                best_sim = -1.0
                best_tgt_idx = -1

                tgt_names = label_space_meta[tgt_id]
                for tgt_idx, tgt_name in enumerate(tgt_names):
                    if valid_labels_dict is not None and tgt_id in valid_labels_dict and tgt_idx not in valid_labels_dict[tgt_id]:
                        continue

                    imgs_tgt = get_images_func(tgt_id, tgt_idx, **get_images_kwargs) if get_images_kwargs else get_images_func(tgt_id, tgt_idx)
                    if imgs_tgt is None: continue

                    proto_tgt = imgs_tgt.view(imgs_tgt.size(0), -1).mean(dim=0).unsqueeze(0)

                    sim = F.cosine_similarity(proto_src, proto_tgt, dim=1).item()

                    if sim >= best_sim:
                        best_sim = sim
                        best_tgt_idx = tgt_idx

                if best_sim >= cs_threshold:
                    mapped_edges.append((src_id, l_idx, tgt_id, best_tgt_idx, best_sim))
                    logger.log(f"  [CS Candidate] {src_id}:'{l_idx}' -> {tgt_id}:'{best_tgt_idx}' (Cosine Similarity: {best_sim:.2f})")

    local_id_to_global_id = {d_id: {} for d_id in dataset_ids}

    sorted_edges = sorted(mapped_edges, key=lambda x: x[4], reverse=True)

    for src_id, l_idx, tgt_id, pred_idx, sim in sorted_edges:
        register_mapping(local_id_to_global_id, src_id, l_idx, tgt_id, pred_idx)

    for d_id in dataset_ids:
        for l_idx in range(len(label_space_meta[d_id])):
            if valid_labels_dict is not None and d_id in valid_labels_dict and l_idx not in valid_labels_dict[d_id]:
                continue
            if l_idx not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                local_id_to_global_id[d_id][l_idx] = current_max + 1

    return local_id_to_global_id   


def global_to_local_mapping(local_id_to_global_id, logger=None, label_space_meta=None):
    global_to_local = {}
    
    for src_id, labels_map in local_id_to_global_id.items():
        for local_id, global_id  in labels_map.items():
            if global_id not in global_to_local:
                global_to_local[global_id] = []
            
            global_to_local[global_id].append((src_id, local_id))
            
    log_func = logger.log if logger else print
    log_func("\n" + "="*50)
    log_func("Label Mapping Summary:")
    
    for gid in sorted(global_to_local.keys()):
        members = global_to_local[gid]
        
        members = sorted(members, key=lambda x: str(x[0]))
        
        member_strs = []
        for src_id, lid in members:
            class_name = label_space_meta[src_id][lid]
            display_str = f"'{class_name}'"
            member_strs.append(f"{src_id}: {display_str}")

        msg = f"Global ID = {gid:<2} | from: " + ", ".join(member_strs)
        log_func(msg)
        
    log_func("="*50 + "\n")
        
    return global_to_local





