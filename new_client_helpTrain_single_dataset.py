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
from utils.utils import initialize_new_clients
from server import Server
from utils.plotting import plot_new_client_accuracy
from models.hetero_model import get_heterogeneous_model, ConditionalGenerator, Classifier
  
DATA_ROOT = './data/raw'

# data input parameter of model & model architecture list 
DATASET_META = { 
    'MNIST':        {'in_ch': 1, 'classes': 10,  'size': 28},  
    'FashionMNIST': {'in_ch': 1, 'classes': 10,  'size': 28},
    'EMNIST':       {'in_ch': 1, 'classes': 47,  'size': 28},
    'CIFAR10':      {'in_ch': 3, 'classes': 10,  'size': 32},
    'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32},
    'CIFAR100_SUPER':{'in_ch': 3, 'classes': 20,  'size': 32},
    'TinyImageNet': {'in_ch': 3, 'classes': 200, 'size': 64} 
}

MODEL_LIST = {
    0: 'MLP', 1: 'CNN', 2: 'ResNet8', 3: 'ResNet18',
    4: 'MobileNetV2', 5: 'MobileNetV3', 6: 'LeNet',
    7: 'AlexNet', 8: 'ShuffleNet', 9: 'SqueezeNet'
}

 
def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    print(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device, weights_only=False)

def find_most_similar_feature(local_feat, gen_feats):
    local_feat_norm = F.normalize(local_feat, dim=1)
    gen_feats_norm = F.normalize(gen_feats, dim=1)
    
    similarity = torch.mm(local_feat_norm, gen_feats_norm.t())
    best_idx = similarity.argmax(dim=1)
    best_gen_feat = gen_feats[best_idx]
    
    return best_gen_feat

def train_new_client_with_generator(client, generator, global_classifier, label_to_global_id, args, logger, num_gen_samples=10):
    generator.eval()
    global_classifier.eval()
    
    client.model.to(args.device)
    
    optimizer = torch.optim.Adam(client.model.parameters(), lr=args.client_lr)
    
    acc_history = []
    acc_history.append(client.test())
    
    for epoch in range(args.new_client_epochs):
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_distill_loss = 0
        
        client.model.train()
        for imgs, labels in client.train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            batch_size = imgs.size(0)
            
            optimizer.zero_grad()
            
            local_feats, logits = client.model(imgs)
            
            # Classification Loss
            loss_ce = nn.CrossEntropyLoss(logits, labels)
            
            # Feature Distillation
            loss_distill = torch.tensor(0.0, device=args.device)
            
            with torch.no_grad():
                best_gen_feats_list = []
                for i in range(batch_size):
                    label = labels[i].item()
                    local_feat = local_feats[i:i+1] 
                    
                    global_label = label_to_global_id.get(label, label)
                    
                    z = torch.randn(num_gen_samples, args.noise_dim, device=args.device)
                    label_input = torch.full((num_gen_samples,), global_label, dtype=torch.long, device=args.device)
                    gen_feats = generator(z, label_input) 
                    
                    best_gen_feat = find_most_similar_feature(local_feat, gen_feats) 
                    best_gen_feats_list.append(best_gen_feat)
                
                teacher_features = torch.cat(best_gen_feats_list, dim=0)
            
            loss_distill = F.mse_loss(local_feats, teacher_features)
            
            total_loss = loss_ce + args.distill_weight * loss_distill
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_distill_loss += loss_distill.item()
        
        acc = client.test()
        acc_history.append(acc)
        logger.log(f"    Epoch {epoch+1}/{args.new_client_epochs} | Loss: {epoch_loss:.4f} (CE: {epoch_ce_loss:.4f}, Distill: {epoch_distill_loss:.4f}) | Acc: {acc:.2f}%")
    
    return acc_history

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
    sub_dir_name = f"new_client_{total_start_time}"
    save_dir = os.path.join(parent_dir, sub_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(args, mode="new_client_single_dataset")
    logger.log("\n" + "="*100, print_to_console=False)
    logger.log(f"   New Clients Training from with Generator Started")
    logger.log(f"   Target Model: {args.model_path}")
    logger.log(f"   Output Dir:   {save_dir}")
    logger.log(f"{'='*100}\n")

    # Load datasets
    all_client_data_loaders = load_partitioned_datasets(args, DATA_ROOT)
    
    # Initialize Server
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

    # Initializing Client
    new_clients, id_to_dataset = initialize_new_clients(
        all_client_data_loaders, DATASET_META, MODEL_LIST, args, DATA_ROOT
    )

    # =========================================
    # New client Training Loop
    # =========================================
    logger.log("\n" + "="*100)
    logger.log("--- Start New Client Training with Generator ---")
    logger.log("="*100)
    for client in new_clients:
        d_name = id_to_dataset[client.client_id]
        d_meta = DATASET_META[d_name]
        client_curves = {}

        full_class_names = get_readable_class_names(d_name, DATA_ROOT)
        ls_id = server._get_label_space_id(full_class_names)
        
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

        for arch_id in range(10):
            arch_name = MODEL_LIST[arch_id]
            logger.log(f"\n  [{arch_name}] Training...")
            model = get_heterogeneous_model(
                client_id=arch_id, 
                in_channels=d_meta['in_ch'],
                num_classes=d_meta['classes'],
                img_size=d_meta['size'],
                global_dim=args.global_feature_dim
            )
            client.model = model

            # Train with generator assistance
            acc_history = train_new_client_with_generator(
                client=client,
                generator=generator,
                global_classifier=global_classifier,
                label_to_global_id=label_to_global_id,
                args=args,
                logger=logger,
                num_gen_samples=args.num_local_noise
            )

            client_curves[arch_id] = [round(float(a), 2) for a in acc_history]
            logger.log(f"      [{arch_name}] History: {client_curves[arch_id]}")

        plot_new_client_accuracy(client_curves, args, d_name, MODEL_LIST, save_dir=logger.get_log_dir())

    # End Messages
    total_duration = time.time() - total_start_time
    logger.log(f"=== Finished. Total Time: {str(timedelta(seconds=int(total_duration)))} ===")

if __name__ == "__main__": 
    main()

    


