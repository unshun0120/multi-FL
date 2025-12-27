import os
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

from config import get_config
from utils.seed import set_seed
from utils.logger import Logger
from data.datasets import load_partitioned_datasets, get_readable_class_names
from server import Server
from utils.plotting import plot_new_client_accuracy
from models.hetero_model import get_heterogeneous_model, ConditionalGenerator, Classifier
from utils.test_utils import initialize_new_clients

DATA_ROOT = './data/raw'

def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    print(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device, weights_only=False)

def find_most_similar_feature(local_feat, gen_feats):
    """
    從 generator 產生的多個 feature 中找最相似的一個
    local_feat: [batch_size, feature_dim]
    gen_feats: [num_classes * num_samples, feature_dim]
    return: best_gen_feat [batch_size, feature_dim], best_indices [batch_size]
    """
    local_feat_norm = F.normalize(local_feat, dim=1)
    gen_feats_norm = F.normalize(gen_feats, dim=1)
    
    similarity = torch.mm(local_feat_norm, gen_feats_norm.t())
    best_indices = similarity.argmax(dim=1)
    best_gen_feats = gen_feats[best_indices]
    
    return best_gen_feats, best_indices

def train_new_client_baseline(client, args, logger):
    """
    Baseline 訓練: 只做 classification loss，不使用 generator
    """
    device = args.device
    client.model.to(device)
    client.model.train()
    
    optimizer = torch.optim.Adam(client.model.parameters(), lr=args.client_lr)
    criterion = nn.CrossEntropyLoss()
    
    acc_history = []
    acc_history.append(client.test())
    
    for epoch in range(args.new_client_epochs):
        epoch_loss = 0
        client.model.train()
        for imgs, labels in client.train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, logits = client.model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        acc = client.test()
        acc_history.append(acc)
        logger.log(f"    Epoch {epoch+1}/{args.new_client_epochs} | Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")
    
    return acc_history

def predict_with_generator_ensemble(
    client,
    generator,
    global_classifier,
    label_to_global_id,
    args,
    num_gen_samples=10,
    lamda=0.5
):
    """
    使用 generator 幫助新 client 預測
    pred = (1 - lamda) * local_pred + lamda * gen_pred
    
    流程:
    1. Generator 預先為所有類別產生 features
    2. 預測時，local feature 找最相似的 generator feature
    3. 最相似的 generator feature 經過 global classifier 得到預測
    4. Ensemble: (1-lamda)*local_prob + lamda*gen_prob
    """
    device = args.device
    generator.eval()
    global_classifier.eval()
    client.model.eval()
    client.model.to(device)
    
    num_classes = len(label_to_global_id)
    
    # 預先為所有類別產生 generator features
    all_gen_feats = []
    
    with torch.no_grad():
        for local_label in range(num_classes):
            global_label = label_to_global_id.get(local_label, local_label)
            
            z = torch.randn(num_gen_samples, args.noise_dim, device=device)
            label_input = torch.full((num_gen_samples,), global_label, dtype=torch.long, device=device)
            gen_feats = generator(z, label_input)  # [num_gen_samples, D]
            
            all_gen_feats.append(gen_feats)
        
        # [num_classes * num_gen_samples, D]
        all_gen_feats = torch.cat(all_gen_feats, dim=0)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in client.test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)
            
            # Local model prediction
            local_feats, local_logits = client.model(imgs)
            local_probs = F.softmax(local_logits, dim=1)
            
            # 找最相似的 generator feature
            best_gen_feats, _ = find_most_similar_feature(local_feats, all_gen_feats)
            
            # Generator feature 經過 global classifier 預測
            gen_logits = global_classifier(best_gen_feats)
            gen_probs = F.softmax(gen_logits, dim=1)
            
            # Ensemble prediction
            ensemble_probs = (1 - lamda) * local_probs + lamda * gen_probs
            preds = ensemble_probs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += batch_size
    
    accuracy = 100.0 * correct / total
    return accuracy

def main():
    total_start_time = time.time()
    args = get_config()
    set_seed(args.seed)

    # =================================
    # Load generator checkpoint
    # =================================
    checkpoint = load_checkpoint(args.model_path, args.device)

    # Create folder to save training result
    parent_dir = os.path.dirname(args.model_path) or '.'
    sub_dir_name = f"new_client_predict_{int(total_start_time)}"
    save_dir = os.path.join(parent_dir, sub_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(args, mode="new_client_predict")
    logger.log(f"{'='*100}")
    logger.log(f"   New Client Prediction with Generator Ensemble")
    logger.log(f"   Checkpoint: {args.model_path}")
    logger.log(f"   Output Dir: {save_dir}")
    logger.log(f"   Lambda: {args.pred_lambda}")
    logger.log(f"{'='*100}\n")

    # Load datasets
    all_client_data_loaders = load_partitioned_datasets(args, DATA_ROOT)
    
    dataset_meta = { 
        'MNIST':        {'in_ch': 1, 'classes': 10,  'size': 28}, 
        'FashionMNIST': {'in_ch': 1, 'classes': 10,  'size': 28},
        'EMNIST':       {'in_ch': 1, 'classes': 47,  'size': 28},
        'CIFAR10':      {'in_ch': 3, 'classes': 10,  'size': 32},
        'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32},
        'CIFAR100_SUPER':{'in_ch': 3, 'classes': 20,  'size': 32},
    }
    
    model_list = {
        0: 'MLP', 1: 'CNN', 2: 'ResNet8', 3: 'ResNet18',
        4: 'MobileNetV2', 5: 'MobileNetV3', 6: 'LeNet',
        7: 'AlexNet', 8: 'ShuffleNet', 9: 'SqueezeNet'
    }
    
    # Initialize Server (for label space utilities)
    server = Server(args)
    server.label_to_global_id = checkpoint['label_to_global_id']
    server.global_id_to_label = checkpoint['global_id_to_label']
    
    # Load generator
    num_global_classes = len(checkpoint['label_to_global_id'])
    generator = ConditionalGenerator(
        num_global_classes=num_global_classes,
        noise_dim=args.noise_dim,
        output_dim=args.global_feature_dim
    ).to(args.device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    # Initializing New Clients
    new_clients, id_to_dataset = initialize_new_clients(
        all_client_data_loaders, 
        dataset_meta, 
        model_list, 
        args, 
        DATA_ROOT
    )

    # =========================================
    # New client Prediction Loop
    # =========================================
    logger.log("\n" + "="*100)
    logger.log("--- Start New Client Prediction with Generator ---")
    logger.log("="*100)

    # 儲存所有結果
    all_results = defaultdict(dict)

    for client in new_clients:
        d_name = id_to_dataset[client.client_id]
        d_meta = dataset_meta[d_name]
        full_class_names = get_readable_class_names(d_name, DATA_ROOT)
        
        # 取得該資料集的 label space id
        ls_id = server._get_label_space_id(full_class_names)
        
        # 檢查是否有對應的 classifier
        if ls_id not in checkpoint['dataset_classifiers']:
            logger.log(f"[Skip] No classifier found for {d_name}")
            continue
        
        # Load global classifier for this dataset
        global_classifier = Classifier(
            input_dim=args.global_feature_dim,
            num_classes=d_meta['classes']
        ).to(args.device)
        global_classifier.load_state_dict(checkpoint['dataset_classifiers'][ls_id])
        global_classifier.eval()
        
        # Build label mapping: local_label -> global_label
        label_to_global_id = {}
        for local_idx, class_name in enumerate(full_class_names):
            if class_name in checkpoint['label_to_global_id']:
                label_to_global_id[local_idx] = checkpoint['label_to_global_id'][class_name]

        logger.log(f"\n{'='*50}")
        logger.log(f"Dataset: {d_name} | Client ID: {client.client_id}")
        logger.log(f"{'='*50}")

        client_curves = {}

        for arch_id in range(10):
            arch_name = model_list[arch_id]
            logger.log(f"\n  [{arch_name}]")
            
            model = get_heterogeneous_model(
                client_id=arch_id, 
                in_channels=d_meta['in_ch'],
                num_classes=d_meta['classes'],
                img_size=d_meta['size'],
                global_dim=args.global_feature_dim
            )
            client.model = model
            client.model_name = arch_name

            # Step 1: Train baseline (只做 classification loss)
            logger.log("  Training (baseline)...")
            acc_history = train_new_client_baseline(client, args, logger)
            baseline_acc = acc_history[-1]

            # Step 2: Predict with generator ensemble
            logger.log("  Predicting with generator ensemble...")
            ensemble_acc = predict_with_generator_ensemble(
                client=client,
                generator=generator,
                global_classifier=global_classifier,
                label_to_global_id=label_to_global_id,
                args=args,
                num_gen_samples=args.num_local_noise,
                lamda=args.pred_lambda
            )

            improvement = ensemble_acc - baseline_acc
            logger.log(f"  Baseline Acc: {baseline_acc:.2f}% | Ensemble Acc: {ensemble_acc:.2f}% | Improvement: {improvement:+.2f}%")

            client_curves[arch_id] = [round(float(a), 2) for a in acc_history]
            logger.log(f"  [{arch_name}] History: {client_curves[arch_id]}", print_to_console=False)

            # 儲存結果
            all_results[d_name][arch_name] = {
                'baseline': baseline_acc,
                'ensemble': ensemble_acc,
                'improvement': improvement
            }

        plot_new_client_accuracy(client_curves, args, d_name, model_list, save_dir=logger.get_log_dir())

    # =========================================
    # Summary
    # =========================================
    logger.log(f"\n{'='*100}")
    logger.log("Summary: Baseline vs Ensemble Prediction")
    logger.log(f"{'='*100}")
    
    for d_name, arch_results in all_results.items():
        logger.log(f"\n{d_name}:")
        avg_baseline = 0
        avg_ensemble = 0
        avg_improvement = 0
        
        for arch_name, acc_dict in arch_results.items():
            logger.log(f"  {arch_name:12s}: Baseline {acc_dict['baseline']:5.2f}% -> Ensemble {acc_dict['ensemble']:5.2f}% ({acc_dict['improvement']:+.2f}%)")
            avg_baseline += acc_dict['baseline']
            avg_ensemble += acc_dict['ensemble']
            avg_improvement += acc_dict['improvement']
        
        num_archs = len(arch_results)
        if num_archs > 0:
            logger.log(f"  {'Average':12s}: Baseline {avg_baseline/num_archs:5.2f}% -> Ensemble {avg_ensemble/num_archs:5.2f}% ({avg_improvement/num_archs:+.2f}%)")

    # End Messages
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    formatted_total_time = str(timedelta(seconds=int(total_duration)))
    logger.log(f"\n=== Finished. Total Time: {formatted_total_time} ===")

if __name__ == "__main__": 
    main()