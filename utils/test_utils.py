import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from data.datasets import get_readable_class_names
from models.hetero_model import get_heterogeneous_model
from client import Client
from utils.plotting import plot_accuracy_curves
from models.hetero_model import ConditionalGenerator

def train_with_generator(client, generator, label_to_global_id, args, logger):
    """使用 generator 輔助訓練 new client"""
    generator.eval()
    client.model.to(args.device)
    
    optimizer = torch.optim.Adam(client.model.parameters(), lr=args.client_lr)
    criterion_ce = nn.CrossEntropyLoss()
    
    acc_history = [client.test()]
    
    for epoch in range(args.new_client_epochs):
        client.model.train()
        epoch_ce, epoch_distill = 0, 0
        
        for imgs, labels in client.train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            local_feats, logits = client.model(imgs)
            
            # Classification Loss
            loss_ce = criterion_ce(logits, labels)
            
            # Feature Distillation Loss
            teacher_feats = get_best_gen_features(local_feats, labels, generator, label_to_global_id, args)
            loss_distill = F.mse_loss(local_feats, teacher_feats)
            
            # Total Loss
            total_loss = loss_ce + args.distill_weight * loss_distill
            total_loss.backward()
            optimizer.step()
            
            epoch_ce += loss_ce.item()
            epoch_distill += loss_distill.item()
        
        acc = client.test()
        acc_history.append(acc)
        logger.log(f"    Epoch {epoch+1}/{args.new_client_epochs} | CE: {epoch_ce:.4f} | Distill: {epoch_distill:.4f} | Acc: {acc:.2f}%")
    
    return acc_history


# Build label mapping: local_label -> global_label
def build_label_mapping(class_names, checkpoint):
    label_to_global_id = {}
    for local_idx, class_name in enumerate(class_names):
        if class_name in checkpoint['label_to_global_id']:
            label_to_global_id[local_idx] = checkpoint['label_to_global_id'][class_name]

    return label_to_global_id


def load_generator(checkpoint, args):
    num_global_classes = len(checkpoint['label_to_global_id'])
    generator = ConditionalGenerator(
        num_global_classes=num_global_classes,
        noise_dim=args.noise_dim,
        output_dim=args.global_feature_dim
    ).to(args.device)

    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    return generator


def find_most_similar_feature(local_feat, gen_feats):
    local_feat_norm = F.normalize(local_feat, dim=1)
    gen_feats_norm = F.normalize(gen_feats, dim=1)
    
    similarity = torch.mm(local_feat_norm, gen_feats_norm.t())
    best_idx = similarity.argmax(dim=1)
    best_gen_feat = gen_feats[best_idx]
    
    return best_gen_feat


def get_best_gen_features(local_feats, labels, generator, label_to_global_id, args):
    """為每個 local feature 找到最相似的 generator feature"""
    batch_size = local_feats.size(0)
    best_gen_feats_list = []
    
    with torch.no_grad():
        for i in range(batch_size):
            local_feat = local_feats[i:i+1]
            global_label = label_to_global_id.get(labels[i].item(), labels[i].item())
            
            z = torch.randn(args.num_local_noise, args.noise_dim, device=args.device)
            label_input = torch.full((args.num_local_noise,), global_label, dtype=torch.long, device=args.device)
            gen_feats = generator(z, label_input)
            
            best_gen_feat = find_most_similar_feature(local_feat, gen_feats)
            best_gen_feats_list.append(best_gen_feat)
    
    return torch.cat(best_gen_feats_list, dim=0)


def initialize_new_clients(all_client_data_loaders, dataset_meta, model_list, args, data_root):
    print("Initializing New Clients...")
    new_clients = []
    id_to_dataset = {}  
    client_id_counter = 0   

    for d_name, loaders_list in all_client_data_loaders.items():
        d_meta = dataset_meta.get(d_name)
        full_class_names = get_readable_class_names(d_name, data_root)
        num_train_client = len(loaders_list) - args.num_new_clients

        for idx, loader_dict in enumerate(loaders_list):
            if idx >= num_train_client:
                train_loader = loader_dict['train']
                test_loader = loader_dict['test']

                client = Client(
                    client_id=client_id_counter,
                    args=args,
                    train_dataset=train_loader.dataset,
                    test_dataset=test_loader.dataset,
                    class_names=full_class_names
                )

                new_clients.append(client)
                id_to_dataset[client_id_counter] = d_name

            client_id_counter += 1

    print(f"Total New Clients Initialized: {len(new_clients)}")

    return new_clients, id_to_dataset


def load_checkpoint(path, device, logger):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    logger.log(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device, weights_only=False)