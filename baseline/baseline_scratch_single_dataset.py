import os
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

from config import get_config
from utils.seed import set_seed
from utils.logger import Logger
from data.datasets import load_partitioned_datasets
from utils.test_utils import initialize_new_clients
from server import Server
from utils.plotting import plot_new_client_accuracy
from models.hetero_model import get_heterogeneous_model

  
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
 

def main():
    total_start_time = time.time()
    args = get_config()
    set_seed(args.seed)
    logger = Logger(args, mode="baseline_scratch")

    # Start Messages
    logger.log("=== Baseline: New Clients Training from Scratch (No Generator) Started ===") 
    logger.log(f"{'='*100}")

    # Preparing & Loading dataset
    # 'MNIST': [ {'train': DataLoader, 'test': DataLoader}, {'train': DataLoader, 'test': DataLoader}, ... ],
    # 'CIFAR10': [ ... ], ...
    all_client_data_loaders = load_partitioned_datasets(args, DATA_ROOT)

    # Initializing Client
    new_clients, id_to_dataset = initialize_new_clients(
        all_client_data_loaders, DATASET_META, MODEL_LIST, args, DATA_ROOT
    )

    # =========================================
    # New client Training Loop
    # =========================================
    logger.log("\n" + "="*100)
    logger.log("--- Start New client Training Loop ---")
    logger.log("="*100)

    for client in new_clients:
        d_name = id_to_dataset[client.client_id]
        d_meta = DATASET_META[d_name]
        client_curves = {}

        for arch_id in range(10):
            arch_name = MODEL_LIST[arch_id]
            model = get_heterogeneous_model(
                client_id=arch_id, 
                in_channels=d_meta['in_ch'],
                num_classes=d_meta['classes'],
                img_size=d_meta['size'],
                global_dim=args.global_feature_dim
            )
            client.model = model

            acc_history = []
            acc_history.append(client.test())
            for epoch in tqdm(range(args.new_client_epochs)):
                client.model.train()
                for imgs, labels in client.train_loader:
                    imgs, labels = imgs.to(args.device), labels.to(args.device)
                    client.optimizer.zero_grad()
                    _, logits = client.model(imgs)
                    loss = client.criterion(logits, labels)
                    loss.backward()
                    client.optimizer.step()
                acc_history.append(client.test())

            msg = f"   [{arch_name}] Final {acc_history:.2f}%"
            logger.log(msg)

            client_curves[arch_id] = [round(float(a), 2) for a in acc_history]
            logger.log(f"      [{arch_name}] History: {client_curves[arch_id]}", print_to_console=False)

        plot_new_client_accuracy(client_curves, args, d_name, MODEL_LIST, save_dir=logger.get_log_dir())

    # End Messages
    total_duration = time.time() - total_start_time
    logger.log(f"=== Baseline (Scratch) Finished. Total Time: {str(timedelta(seconds=int(total_duration)))} ===")

if __name__ == "__main__": 
    main()

    


